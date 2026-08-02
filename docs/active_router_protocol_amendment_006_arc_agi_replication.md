# Amendment 006: untouched ARC-AGI sequential replication

Status: **frozen before aggregate ARC-AGI outcome analysis**

Effective date: 2026-07-29

Parent study: `open-router-study-2026-07-29-v1`

## Reason and inferential status

The confirmatory public-study primary policy failed to preserve fixed-model
quality on the ID test. A prespecified secondary curve point—the shared
task-model classifier at cost weight `0.10`—matched the calibration-best
fixed model while reducing cost. The reusable implementation was then made to
reproduce that exact frozen classifier and calibration behavior.

This amendment adds one untouched public dataset as a **sequential external
replication**. It does not replace or repair the failed confirmatory primary
analysis. The choice to test this classifier was informed by ID results, so
the replication is explicitly follow-up evidence rather than a new
confirmatory primary result.

Before this freeze, one ARC-AGI example and file-level metadata had been
inspected during source validation. No aggregate ARC-AGI outcomes or
model-to-model ARC-AGI comparisons were calculated.

## Frozen source and artifact

- Source dataset and revisions are identical to Amendment 005.
- Dataset: `arc-agi`
- Split: `v1` (the release's ARC-AGI evaluation split).
- Candidate models: the same 13 flagship models in Amendment 005.
- Deployable artifact SHA-256 identity:
  `677eba863b0d1eb765560e31992bfdcf94364325d1e11f08468d7110c35ab0d8`
- Classifier: shared task-model hashed logistic classifier.
- Hash dimension: `65536`.
- Training seed: `20260729`.
- Cost weight: `0.10`.
- Calibration: per-model isotonic regression learned only from the original
  calibration split.

The artifact, model weights, calibration functions, candidate pool, cost
estimates, and selection weight are immutable for this replication. ARC-AGI
outcomes may not be used for fitting, calibration, threshold selection, or
model-pool changes.

## Inclusion and evaluation

Use only ARC-AGI prompt keys with a valid and matching result for all 13
candidate models. Exclude incomplete, duplicate, mismatched, non-finite, or
negative-cost rows and report exclusion counts. Do not substitute another
model or timestamped result file.

Primary comparison:

- frozen router versus the calibration-selected fixed model, `gpt-5`;
- paired mean score difference;
- total realized cost and fractional cost saving.

Also report:

- every fixed model;
- the post-hoc ARC-AGI-best fixed model, clearly labeled descriptive;
- binary acceptable rate at score `>= 0.5`;
- route counts and entropy;
- paired prompt bootstrap 95% intervals for score difference and cost saving,
  using seed `20260729` and 10,000 resamples.

No additional paid inference is authorized or required for this replication.
