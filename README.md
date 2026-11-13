---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-68e59a5ac91ad404**

[Dataset website](https://earino.github.io/factory-fda-510k-five-year-recall/versions/v-68e59a5ac91ad404/) · [GitHub release](https://github.com/earino/factory-fda-510k-five-year-recall/releases/tag/v-68e59a5ac91ad404) · [Hugging Face](https://huggingface.co/datasets/earino/factory-fda-510k-five-year-recall/tree/v-68e59a5ac91ad404) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: 4402989d50d250e5d9eab6a8f93c00997c3be07ffecedd577a04ceddd1c15d15
<!-- /dataset-factory:links -->

# Five-year post-market recall of U.S. medical devices cleared through the 510(k) premarket notification pathway

Will a medical device cleared by FDA through the 510(k) pathway appear in FDA's device recall enforcement reports, for a recall initiated within five years of the clearance decision?

Derived from U.S. Food and Drug Administration (openFDA, Center for Devices and Radiological Health). This is not its official product.

Prediction time: The clearance decision date recorded on the 510(k) record (decision_date). Only fields on that record plus counts of strictly earlier clearance dates are used.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6798441538385072,
    "brier": 0.0794854534527989
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.08262498077057921
  },
  "calendar_identity": {
    "auc": 0.6060407436072217,
    "brier": 0.09564997028543741
  }
}
```

## Limitations

- The outcome is linked only through the 510(k) number: 20,792 of 59,705 device recall records (about 35%) carry no k_number and are invisible, so the label is a lower bound on true recall incidence for cleared devices.
- One recall event frequently names several 510(k) numbers and one firm recurs across many rows, so positives are correlated within recall events and within firms; effective sample size for positives is smaller than the row count suggests.
- event_date_initiated is the firm's action date, not FDA's classification date, so recalls initiated near the end of a five-year window but classified later can be miscounted at the boundary.
- device_class comes from openFDA's harmonised join against FDA's current classification database, so it can differ from the class in force at clearance (current-vintage limitation, stated rather than hidden).
- Applicant strings are free text, so the same firm appears under spelling variants and applicant_prior_clearances undercounts large filers.
- The 510(k) record does not say whether the device was ever marketed; an unknown share of 'no recall' rows are devices that never reached patients, which is a definitional limitation of the label, not a data error.
- The bulk files are a moving current vintage (FDA notes later updates can change old records), so the exact row set is reproducible only by archiving the two zips and their checksums.
- 510(k) records carry a named contact person and street address; those columns are dropped before release, and no individual-level field is published in the derived CSVs. Only the 510k and recall endpoints are used, so the GMDN-restricted content noted on the licence page is out of scope.
- The outcome is linked only through the 510(k) number: 20,792 of 59,705 device recall product records carry no k_number and are invisible, so the label is a lower bound on true recall incidence for cleared devices.
- event_date_initiated is the firm's action date, not FDA's classification date, so a recall initiated near the end of a window but classified later can sit at the boundary.
- device_class comes from openFDA's harmonised join against the current classification database, so it can differ from the class in force at clearance.
- The 510(k) record does not say whether the device was ever marketed; an unknown share of no-recall rows are devices that never reached patients.
- When training labels became available is not documented by the publisher, so the label timing is recorded as unverified rather than as a forecasting claim.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
