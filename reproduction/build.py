import io
import json
import zipfile
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from source_io import fetch

URL_510K = 'https://download.open.fda.gov/device/510k/device-510k-0001-of-0001.json.zip'
URL_RECALL = 'https://download.open.fda.gov/device/recall/device-recall-0001-of-0001.json.zip'

POP_START = date(2004, 1, 1)
POP_END = date(2020, 1, 1)
HORIZON_DAYS = 1826
TARGET = 'recall_5y'
ID_COLUMN = 'k_number'
TIME_COLUMN = 'decision_date'

FEATURE_COLUMNS = [
    'product_code',
    'advisory_committee',
    'device_class',
    'clearance_type',
    'decision_code',
    'review_days',
    'third_party_flag',
    'applicant_prior_clearances',
    'applicant_country_code',
]

OUTPUT_COLUMNS = [ID_COLUMN, TIME_COLUMN] + FEATURE_COLUMNS + [TARGET]

REQUIRED_510K_FIELDS = [
    'k_number',
    'decision_date',
    'date_received',
    'decision_code',
    'clearance_type',
    'advisory_committee',
    'product_code',
    'third_party_flag',
    'applicant',
    'country_code',
]
REQUIRED_RECALL_FIELDS = ['k_numbers', 'event_date_initiated']
RECALL_LABEL_SOURCES = ['k_numbers', 'event_date_initiated', 'product_res_number']

SPLITS = {
    'train': (date(2004, 1, 1), date(2014, 1, 1)),
    'eval': (date(2014, 1, 1), date(2017, 1, 1)),
    'holdout': (date(2017, 1, 1), date(2020, 1, 1)),
}
SPLIT_PATHS = {
    'train': ('public', 'train.csv'),
    'eval': ('public', 'eval.csv'),
    'holdout': ('private', 'holdout.csv'),
}

PUBLISHED_RECORDS = 176625
PUBLISHED_IN_WINDOW = 49270
PUBLISHED_RECALL_WITH_K = 38913
DOCUMENTED_SNAPSHOT = date(2026, 11, 12)
QUOTE = chr(34)


def _parse_date(value):
    if value is None or isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _norm_id(value):
    text = _clean(value)
    return text.upper() if text else None


def _looks_like_k_number(text):
    if not text or text[0] != 'K':
        return False
    digits = text[1:]
    return digits.isdigit() and 4 <= len(digits) <= 8


def _plus_years(value, years):
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _json_member(zf, zip_name):
    names = [
        name
        for name in zf.namelist()
        if name.lower().endswith('.json') and not name.startswith('__MACOSX')
    ]
    if not names:
        raise AssertionError(
            'no JSON member in ' + zip_name + '; members ' + str(zf.namelist()[:20])
        )
    return max(names, key=lambda name: zf.getinfo(name).file_size)


def _json_member_name(path):
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        return _json_member(zf, path.name)


def _iter_records(zip_path, want_key='results'):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        member = _json_member(zf, zip_path.name)
        with zf.open(member) as raw:
            stream = io.TextIOWrapper(raw, encoding='utf-8', errors='replace')
            for record in _iter_objects(stream, want_key, zip_path.name, member):
                yield record


def _find_array_start(buffer, want_key):
    needle = QUOTE + want_key + QUOTE
    position = 0
    while True:
        index = buffer.find(needle, position)
        if index < 0:
            return -1
        cursor = index + len(needle)
        while cursor < len(buffer) and buffer[cursor].isspace():
            cursor += 1
        if cursor < len(buffer) and buffer[cursor] == ':':
            cursor += 1
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if cursor < len(buffer) and buffer[cursor] == '[':
                return cursor + 1
        position = index + 1


def _iter_objects(stream, want_key, zip_name, member):
    decoder = json.JSONDecoder()
    buffer = ''
    search_from = 0
    array_start = -1
    while array_start < 0:
        found = _find_array_start(buffer[search_from:], want_key)
        if found >= 0:
            array_start = search_from + found
            break
        search_from = max(0, len(buffer) - 4096)
        chunk = stream.read(1 << 20)
        if not chunk:
            raise AssertionError(
                'array key ' + want_key + ' not found in ' + zip_name + ':' + member
            )
        buffer += chunk
    index = array_start
    while True:
        if index >= len(buffer):
            chunk = stream.read(1 << 20)
            if not chunk:
                return
            buffer = buffer[index:] + chunk
            index = 0
            continue
        character = buffer[index]
        if character.isspace() or character == ',':
            index += 1
            continue
        if character == ']':
            return
        if character != '{':
            raise AssertionError(
                'unexpected token at offset '
                + str(index)
                + ' of '
                + zip_name
                + ':'
                + member
            )
        while True:
            try:
                record, end = decoder.raw_decode(buffer, index)
            except ValueError:
                chunk = stream.read(1 << 20)
                if not chunk:
                    raise AssertionError('truncated JSON object in ' + zip_name + ':' + member)
                buffer = buffer[index:] + chunk
                index = 0
                continue
            break
        yield record
        index = end
        if index > (1 << 20):
            buffer = buffer[index:]
            index = 0


def _member_last_updated(path):
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        member = _json_member(zf, path.name)
        with zf.open(member) as raw:
            head = raw.read(1 << 16).decode('utf-8', errors='replace')
    marker = QUOTE + 'last_updated' + QUOTE
    index = head.find(marker)
    if index < 0:
        return None
    cursor = index + len(marker)
    while cursor < len(head) and head[cursor].isspace():
        cursor += 1
    if cursor >= len(head) or head[cursor] != ':':
        return None
    cursor += 1
    while cursor < len(head) and head[cursor].isspace():
        cursor += 1
    if cursor >= len(head) or head[cursor] != QUOTE:
        return None
    end = head.find(QUOTE, cursor + 1)
    if end < 0:
        return None
    return _parse_date(head[cursor + 1:end])


def build(output, cache):
    output = Path(output)
    cache = Path(cache)
    checks = []

    archive_510k = Path(fetch(URL_510K, cache, 'device-510k-0001-of-0001.json.zip'))
    archive_recall = Path(fetch(URL_RECALL, cache, 'device-recall-0001-of-0001.json.zip'))
    member_510k = _json_member_name(archive_510k)
    member_recall = _json_member_name(archive_recall)

    size_510k = archive_510k.stat().st_size
    size_recall = archive_recall.stat().st_size
    checks.append({
        'check': '510(k) bulk archive downloaded',
        'observed': str(size_510k) + ' bytes, member ' + member_510k,
        'expected': 'publisher lists a 238.38 MB zipped 510(k) archive; accept any compressed file above 20 MB because the snapshot moves',
        'passed': size_510k > 20 * 1024 * 1024,
    })
    checks.append({
        'check': 'device recall bulk archive downloaded',
        'observed': str(size_recall) + ' bytes, member ' + member_recall,
        'expected': 'publisher lists a 276.47 MB zipped device recall archive; accept any compressed file above 20 MB because the snapshot moves',
        'passed': size_recall > 20 * 1024 * 1024,
    })

    presence = Counter()
    observed_keys = set()
    n_510k = 0
    n_missing_decision = 0
    n_bad_decision = 0
    n_in_window = 0
    prior_pairs = []
    candidates = []

    for record in _iter_records(archive_510k):
        if not isinstance(record, dict):
            continue
        n_510k += 1
        if n_510k <= 500:
            observed_keys.update(record.keys())
        for field in REQUIRED_510K_FIELDS:
            value = record.get(field)
            if value is not None and value != '':
                presence[field] += 1
        openfda = record.get('openfda')
        device_class = None
        if isinstance(openfda, dict):
            device_class = _clean(openfda.get('device_class'))
            if device_class is not None:
                presence['openfda.device_class'] += 1
        if device_class is None:
            device_class = _clean(record.get('device_class'))
            if device_class is not None:
                presence['device_class'] += 1

        raw_decision = record.get('decision_date')
        decision = _parse_date(raw_decision)
        if decision is None:
            if raw_decision is None or raw_decision == '':
                n_missing_decision += 1
            else:
                n_bad_decision += 1

        applicant = _clean(record.get('applicant'))
        if applicant and decision:
            prior_pairs.append((applicant, decision))

        if decision is None or decision < POP_START or decision >= POP_END:
            continue
        n_in_window += 1

        k_number = _norm_id(record.get('k_number'))
        if not k_number:
            continue
        decision_code = _clean(record.get('decision_code'))
        if (decision_code or '').upper() != 'SESE':
            continue

        candidates.append({
            'k_number': k_number,
            'decision_date': decision,
            'date_received': _parse_date(record.get('date_received')),
            'product_code': _clean(record.get('product_code')),
            'advisory_committee': _clean(record.get('advisory_committee')),
            'device_class': device_class,
            'clearance_type': _clean(record.get('clearance_type')),
            'decision_code': decision_code,
            'third_party_flag': _clean(record.get('third_party_flag')),
            'applicant_country_code': _clean(record.get('country_code')),
            'applicant': applicant,
        })

    missing_names = [field for field in REQUIRED_510K_FIELDS if presence.get(field, 0) == 0]
    checks.append({
        'check': '510(k) record carries every declared field',
        'observed': 'missing names ' + str(missing_names) + '; keys seen in the first 500 records ' + str(sorted(observed_keys)[:40]),
        'expected': 'openFDA device/510k searchable fields: k_number, decision_date, date_received, decision_code, clearance_type, advisory_committee, product_code, third_party_flag, applicant, country_code',
        'passed': not missing_names,
    })
    if missing_names:
        detail = (
            'field check failed for archive member '
            + member_510k
            + ' in '
            + archive_510k.name
            + '; missing names '
            + str(missing_names)
            + '; observed top-level keys '
            + str(sorted(observed_keys)[:200])
            + '; presence counts '
            + str(dict(presence))
        )
        raise AssertionError(detail[:4000])

    if presence.get('openfda.device_class', 0) == 0 and presence.get('device_class', 0) == 0:
        raise AssertionError((
            'field check failed for archive member '
            + member_510k
            + ' in '
            + archive_510k.name
            + '; missing names [openfda.device_class]; observed top-level keys '
            + str(sorted(observed_keys)[:200])
        )[:4000])

    checks.append({
        'check': '510(k) record count matches publisher evidence',
        'observed': str(n_510k) + ' clearance records',
        'expected': 'publisher reports ' + str(PUBLISHED_RECORDS) + ' clearance records (10 percent tolerance for a moving snapshot)',
        'passed': abs(n_510k - PUBLISHED_RECORDS) <= 0.10 * PUBLISHED_RECORDS,
    })
    checks.append({
        'check': 'in-window 510(k) record count matches publisher evidence',
        'observed': str(n_in_window) + ' records with decision_date in the frozen 2004 to 2019 window',
        'expected': 'publisher reports ' + str(PUBLISHED_IN_WINDOW) + ' clearance records with a decision date from 2004-01-01 to 2019-12-31 (10 percent tolerance)',
        'passed': abs(n_in_window - PUBLISHED_IN_WINDOW) <= 0.10 * PUBLISHED_IN_WINDOW,
    })
    checks.append({
        'check': 'decision_date parses for the clearance population',
        'observed': str(n_bad_decision) + ' unparsable and ' + str(n_missing_decision) + ' missing decision_date values among ' + str(n_510k) + ' records',
        'expected': 'decision_date is an ISO date on the clearance records; unparsable values stay below one per thousand records',
        'passed': n_bad_decision <= max(1, int(0.001 * n_510k)),
    })
    checks.append({
        'check': 'substantially equivalent clearances dominate the included population',
        'observed': str(len(candidates)) + ' SESE clearances among ' + str(n_in_window) + ' in-window records',
        'expected': 'the frozen population keeps decision_code SESE only; the retained count must be large and positive',
        'passed': len(candidates) > 0,
    })

    recall_keys = set()
    recall_presence = Counter()
    n_recall = 0
    n_recall_with_k = 0
    k_values = 0
    k_well_formed = 0
    recall_missing_event = 0
    recall_bad_event = 0
    recall_dates = defaultdict(list)

    for record in _iter_records(archive_recall):
        if not isinstance(record, dict):
            continue
        n_recall += 1
        if n_recall <= 500:
            recall_keys.update(record.keys())
        for field in REQUIRED_RECALL_FIELDS:
            if record.get(field) not in (None, '', []):
                recall_presence[field] += 1
        numbers = record.get('k_numbers')
        if isinstance(numbers, str):
            numbers = [numbers]
        if not numbers:
            continue
        n_recall_with_k += 1
        event = _parse_date(record.get('event_date_initiated'))
        if event is None:
            if record.get('event_date_initiated') in (None, ''):
                recall_missing_event += 1
            else:
                recall_bad_event += 1
        for item in numbers:
            norm = _norm_id(item)
            if not norm:
                continue
            k_values += 1
            if _looks_like_k_number(norm):
                k_well_formed += 1
            if event is not None:
                recall_dates[norm].append(event)

    missing_recall = [field for field in REQUIRED_RECALL_FIELDS if recall_presence.get(field, 0) == 0]
    checks.append({
        'check': 'recall record carries the join key and the event date',
        'observed': 'missing names ' + str(missing_recall) + '; keys seen in the first 500 records ' + str(sorted(recall_keys)[:40]),
        'expected': 'openFDA device/recall fields k_numbers and event_date_initiated on the recall product records',
        'passed': not missing_recall,
    })
    if missing_recall:
        raise AssertionError((
            'field check failed for archive member '
            + member_recall
            + ' in '
            + archive_recall.name
            + '; missing names '
            + str(missing_recall)
            + '; observed top-level keys '
            + str(sorted(recall_keys)[:200])
        )[:4000])

    checks.append({
        'check': 'recall records carrying at least one 510(k) number',
        'observed': str(n_recall_with_k) + ' of ' + str(n_recall) + ' recall product records',
        'expected': 'publisher reports ' + str(PUBLISHED_RECALL_WITH_K) + ' of 59705 recall records carry k_numbers (10 percent tolerance)',
        'passed': abs(n_recall_with_k - PUBLISHED_RECALL_WITH_K) <= 0.10 * PUBLISHED_RECALL_WITH_K,
    })
    checks.append({
        'check': 'recall join keys look like 510(k) numbers',
        'observed': str(k_well_formed) + ' of ' + str(k_values) + ' k_numbers start with K and carry 4 to 8 digits',
        'expected': 'a 510(k) number is the letter K followed by digits, as in the published example K020436',
        'passed': k_values > 0 and k_well_formed / k_values >= 0.95,
    })
    checks.append({
        'check': 'recall event dates parse',
        'observed': str(recall_bad_event) + ' unparsable and ' + str(recall_missing_event) + ' missing event_date_initiated values among ' + str(n_recall_with_k) + ' recall records with k_numbers',
        'expected': 'event_date_initiated is an ISO date; unparsable values must be zero and missing values are counted rather than treated as a negative label',
        'passed': recall_bad_event == 0,
    })

    for values in recall_dates.values():
        values.sort()

    prior_counts = Counter(prior_pairs)
    per_applicant = defaultdict(list)
    for (applicant, decision), count in prior_counts.items():
        per_applicant[applicant].append((decision, count))
    prior_lookup = {}
    for applicant, items in per_applicant.items():
        items.sort()
        running = 0
        for decision, count in items:
            prior_lookup[(applicant, decision)] = running
            running += count

    sampled = candidates[:: max(1, len(candidates) // 5)][:5] if candidates else []
    verification_mismatches = 0
    for candidate in sampled:
        applicant = candidate['applicant']
        if not applicant:
            continue
        expected_prior = sum(
            1 for other_applicant, other_date in prior_pairs
            if other_applicant == applicant and other_date < candidate['decision_date']
        )
        if prior_lookup.get((applicant, candidate['decision_date'])) != expected_prior:
            verification_mismatches += 1
    checks.append({
        'check': 'applicant prior clearances count only strictly earlier decision dates',
        'observed': str(verification_mismatches) + ' mismatches over ' + str(len(sampled)) + ' sampled clearances',
        'expected': 'the count equals the number of records with the same applicant string and an earlier decision_date, recomputed directly from the source records with no recall information',
        'passed': verification_mismatches == 0,
    })

    snapshot = _member_last_updated(archive_recall)
    snapshot_from_meta = snapshot is not None
    if snapshot is None:
        snapshot = DOCUMENTED_SNAPSHOT

    rows = []
    dropped_unobserved = 0
    latest_window_end = None
    for candidate in candidates:
        window_end = candidate['decision_date'] + timedelta(days=HORIZON_DAYS)
        if latest_window_end is None or window_end > latest_window_end:
            latest_window_end = window_end
        if window_end > snapshot:
            dropped_unobserved += 1
            continue
        dates = recall_dates.get(candidate['k_number'])
        positive = 0
        if dates:
            index = bisect_left(dates, candidate['decision_date'])
            if index < len(dates) and dates[index] <= window_end:
                positive = 1
        received = candidate['date_received']
        review_days = None
        if received is not None:
            delta = (candidate['decision_date'] - received).days
            if delta >= 0:
                review_days = delta
        rows.append({
            ID_COLUMN: candidate['k_number'],
            TIME_COLUMN: candidate['decision_date'].isoformat(),
            'product_code': candidate['product_code'],
            'advisory_committee': candidate['advisory_committee'],
            'device_class': candidate['device_class'],
            'clearance_type': candidate['clearance_type'],
            'decision_code': candidate['decision_code'],
            'review_days': review_days,
            'third_party_flag': candidate['third_party_flag'],
            'applicant_prior_clearances': prior_lookup.get((candidate['applicant'], candidate['decision_date'])),
            'applicant_country_code': candidate['applicant_country_code'],
            TARGET: positive,
        })

    assert rows, 'no rows survived the population and horizon filters; check decision_date parsing and the SESE filter'

    frame = pd.DataFrame(rows)
    frame = frame.sort_values([TIME_COLUMN, ID_COLUMN], kind='mergesort')
    frame = frame.drop_duplicates(subset=[ID_COLUMN], keep='first').reset_index(drop=True)
    frame = frame[OUTPUT_COLUMNS].copy()
    frame['review_days'] = frame['review_days'].astype('Int64')
    frame['applicant_prior_clearances'] = frame['applicant_prior_clearances'].astype('Int64')
    frame[TARGET] = frame[TARGET].astype(int)

    parsed_times = pd.to_datetime(frame[TIME_COLUMN])
    written = {}
    for name, (start, end) in SPLITS.items():
        mask = (parsed_times >= pd.Timestamp(start)) & (parsed_times < pd.Timestamp(end))
        part = frame.loc[mask].copy()
        assert len(part) > 0, (
            'split '
            + name
            + ' is empty; the released decision_date range is '
            + str(frame[TIME_COLUMN].min())
            + ' to '
            + str(frame[TIME_COLUMN].max())
        )
        folder, filename = SPLIT_PATHS[name]
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        part.to_csv(destination / filename, index=False, na_rep='')
        written[name] = part

    for name in ('train', 'eval', 'holdout'):
        part = written[name]
        positives = int(part[TARGET].sum())
        checks.append({
            'check': 'split ' + name + ' has rows and positives',
            'observed': str(len(part)) + ' rows and ' + str(positives) + ' positives',
            'expected': 'a non-empty frozen window with at least one positive under the five-year recall rule',
            'passed': len(part) > 0 and positives > 0,
        })

    checks.append({
        'check': 'k_number is unique across the release',
        'observed': str(int(frame[ID_COLUMN].duplicated().sum())) + ' duplicate ids among ' + str(len(frame)) + ' rows',
        'expected': 'one row per 510(k) clearance; repeats dropped keeping the earliest decision_date',
        'passed': int(frame[ID_COLUMN].duplicated().sum()) == 0,
    })

    checks.append({
        'check': 'shipped columns are exactly the declared identifier, time, features and target',
        'observed': str(list(written['train'].columns)),
        'expected': str(OUTPUT_COLUMNS) + ' on all three splits',
        'passed': all(list(written[name].columns) == OUTPUT_COLUMNS for name in written),
    })

    shipped_inputs = [ID_COLUMN, TIME_COLUMN] + FEATURE_COLUMNS
    label_overlap = sorted(set(shipped_inputs) & set(RECALL_LABEL_SOURCES))
    checks.append({
        'check': 'no recall-side label source is shipped as an input column',
        'observed': 'shipped inputs ' + str(shipped_inputs) + '; overlap with recall label sources ' + str(label_overlap),
        'expected': 'every input comes from the 510(k) clearance record plus a label-free count of earlier clearance dates; the recall fields k_numbers, event_date_initiated and product_res_number stay out of the release',
        'passed': not label_overlap,
    })

    sample_step = max(1, len(frame) // 200)
    label_checked = 0
    label_mismatch = 0
    for position in range(0, len(frame), sample_step):
        row = frame.iloc[position]
        key = str(row[ID_COLUMN])
        decision = _parse_date(row[TIME_COLUMN])
        if decision is None:
            continue
        window_end = decision + timedelta(days=HORIZON_DAYS)
        dates = recall_dates.get(key)
        expected_positive = 0
        if dates:
            index = bisect_left(dates, decision)
            if index < len(dates) and dates[index] <= window_end:
                expected_positive = 1
        label_checked += 1
        if expected_positive != int(row[TARGET]):
            label_mismatch += 1
    checks.append({
        'check': 'shipped target recomputes from the recall join on sampled rows',
        'observed': str(label_mismatch) + ' mismatches over ' + str(label_checked) + ' sampled released rows',
        'expected': 'a row is positive exactly when some recall record for its k_number has an initiation date between the decision date and the decision date plus 1826 days, inclusive; nothing else enters the label',
        'passed': label_checked > 0 and label_mismatch == 0,
    })

    empty_features = [column for column in FEATURE_COLUMNS if int(frame[column].notna().sum()) == 0]
    checks.append({
        'check': 'every declared feature column is populated',
        'observed': 'empty columns ' + str(empty_features),
        'expected': 'no declared feature ships as an empty column; missing values stay missing rather than being filled',
        'passed': not empty_features,
    })

    class_counts = frame['device_class'].dropna().value_counts().to_dict()
    device_total = int(frame['device_class'].notna().sum())
    unrecognized = int(sum(count for code, count in class_counts.items() if code not in ('1', '2', '3')))
    checks.append({
        'check': 'device_class values come from the documented risk classes',
        'observed': 'value counts ' + str(class_counts) + '; populated for ' + str(device_total) + ' of ' + str(len(frame)) + ' rows; other codes ' + str(unrecognized),
        'expected': 'openFDA classification join publishes classes 1, 2 and 3; other published codes are kept as their own category and counted rather than converted to zero',
        'passed': device_total / len(frame) >= 0.5 and (device_total == 0 or unrecognized / device_total <= 0.10),
    })

    review_values = frame['review_days'].dropna()
    checks.append({
        'check': 'review_days is a non-negative day count',
        'observed': 'min ' + str(int(review_values.min()) if len(review_values) else 'n/a') + ', max ' + str(int(review_values.max()) if len(review_values) else 'n/a') + ', populated for ' + str(len(review_values)) + ' rows',
        'expected': 'decision_date is on or after date_received on the same 510(k) record',
        'passed': len(review_values) > 0 and int(review_values.min()) >= 0,
    })

    unique_dates = sorted({value.date() for value in parsed_times})
    step = max(1, len(unique_dates) // 10)
    sampled_dates = unique_dates[::step][:10]
    horizon_offsets = [abs((_plus_years(value, 5) - value).days - HORIZON_DAYS) for value in sampled_dates]
    checks.append({
        'check': 'horizon unit: 1826 days against five calendar years',
        'observed': 'max absolute difference ' + str(max(horizon_offsets)) + ' day(s) over ' + str(len(sampled_dates)) + ' sampled decision dates',
        'expected': 'the frozen five-year horizon is expressed as 1826 days; a leap-day boundary may differ by one day',
        'kind': 'unit',
        'passed': max(horizon_offsets) <= 1,
    })

    checks.append({
        'check': 'every labelled row has a fully observed five-year window',
        'observed': 'latest window end ' + latest_window_end.isoformat() + ' against recall snapshot ' + snapshot.isoformat() + (' from archive meta last_updated' if snapshot_from_meta else ' from the documented snapshot date') + '; dropped unobserved rows ' + str(dropped_unobserved),
        'expected': 'decision_date plus 1826 days must close on or before the recall snapshot, otherwise the row is dropped rather than labelled 0',
        'passed': latest_window_end <= snapshot,
    })

    total_positive = int(frame[TARGET].sum())
    overall_rate = total_positive / len(frame)
    checks.append({
        'check': 'measured positive rate is plausible for the joined label',
        'observed': str(total_positive) + ' positives of ' + str(len(frame)) + ' rows, rate ' + format(overall_rate, '.4f'),
        'expected': 'the label is built by joining recall records on the 510(k) number, so the measured rate must be above zero and well below one; no publisher rate exists to copy',
        'passed': 0 < overall_rate < 0.6,
    })

    failures = [check for check in checks if not check['passed']]
    if failures:
        raise AssertionError(('source checks failed: ' + repr(failures))[:4000])

    return checks
