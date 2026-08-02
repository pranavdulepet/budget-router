# Amendment 005: pinned public outcome-matrix study

Status: **normative before aggregate outcome analysis**

Effective date: 2026-07-29

Parent study: `open-router-study-2026-07-29-v1`

## Reason

The predeclared Tinker screening gate retained only two model families. The
active protocol therefore stops the paid development and held-out stages and
continues the statistically powered classifier study on LLMRouterBench, as
originally specified.

This amendment freezes the previously unspecified public-data revision,
model pool, exact datasets, splits, labels, and selection rule. Before this
freeze, inspection was limited to release/schema validation and four example
records. The observed examples and their file-level metadata are disclosed in
the final report; no aggregate outcome comparison informed this design.

## Source lock

- GitHub source revision:
  `c77cb0506949d8f959e97967d2fefca0e8ff1b05`
- Hugging Face dataset:
  `NPULH/LLMRouterBench`
- Dataset revision:
  `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`
- Archive:
  `bench-release.tar.gz`
- Archive SHA-256:
  `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`

Only prompts with a complete outcome row for every candidate model are used.
If multiple result files exist for one dataset/split/model, the
lexicographically latest timestamped file is selected and recorded.

## Frozen flagship model pool

The public study uses the 13-model performance-cost pool from the pinned
upstream configuration:

- `claude-sonnet-4`
- `deepseek-v3-0324`
- `deepseek-v3.1-terminus`
- `deepseek-r1-0528`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gpt-5-chat`
- `gpt-5`
- `qwen3-235b-a22b-2507`
- `qwen3-235b-a22b-thinking-2507`
- `glm-4.6`
- `kimi-k2-0905`
- `intern-s1`

These public outcomes are also the reusable frontier fixed-model references.
No fresh frontier API calls are authorized.

## Datasets and splits

In-distribution source datasets:

- `aime/hybrid`
- `livemathbench/test`
- `gpqa/test`
- `hle/test`
- `livecodebench/test`
- `mmlupro/test_3000`
- `swe-bench/verified`
- `simpleqa/test`
- `tau2/test`

For each source dataset, prompt groups are assigned by a deterministic
SHA-256 ordering with seed `open-router-public-v1-20260729`:

- first 60%: training;
- next 20%: calibration/model selection;
- final 20%: untouched ID test.

All rows for the same dataset, record index, and normalized origin query stay
in the same split.

OOD datasets are test-only:

- `arenahard_coding/test`
- `arenahard_math/test`
- `arenahard_creative_writing/test`

The combined `arenahard` directory is excluded because it duplicates those
three partitions. `arc-agi` is excluded because it is outside the pinned
upstream performance-cost dataset list.

## Outcomes and costs

- Quality is the upstream standardized `score`, clipped only for validation
  to the declared range `[0, 1]`.
- A binary acceptable-outcome label is `score >= 0.5`.
- Primary quality is macro mean score across datasets.
- Micro mean score and binary success rate are secondary.
- Realized selected-row `cost` is the primary cost measure.
- Model-card expected cost is learned from training rows only and is used for
  routing; held-out realized cost is never exposed to a deployable selector.
- Rows with non-finite scores, negative costs, mismatched prompts, duplicate
  keys, or incomplete model matrices are excluded with reason counts.

## Required methods

Controls:

- every fixed model;
- cheapest fixed model by training mean cost;
- calibration-best fixed model;
- uniform random;
- frequency-matched random;
- training-only dataset lookup, with global-best fallback for unseen domains;
- hindsight score oracle, labeled non-deployable.

Learned methods:

- independent hashed logistic heads;
- shared task-model hashed classifier;
- text-nearest-neighbor outcome regressor;
- calibrated gradient-boosted task-model classifier.

All learned probability heads receive calibration-only isotonic calibration
per model, falling back to global calibration when a per-model fit is
degenerate.

## Frozen selection and evaluation

Every learned method emits the same selector family:

`predicted_score - lambda * normalized_training_expected_cost`

where `lambda` is in:

`[0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]`.

The primary deployable policy is selected using calibration outcomes only:

1. retain configurations whose macro score is no more than 0.5 percentage
   points below the calibration-best fixed model;
2. choose the lowest macro mean cost;
3. break ties by higher macro score, lower Brier score, and then lower method
   complexity in this order: independent logistic, shared logistic, k-nearest
   neighbor, gradient boosting;
4. if no learned configuration clears the quality floor, choose the
   highest-macro-score learned configuration and label the non-inferiority
   gate failed.

The primary policy and all selector settings are frozen before ID/OOD labels
are scored. Report:

- macro and micro score, binary success, mean/total cost, and cost per
  successful outcome;
- Brier score, expected calibration error, route distribution, and entropy;
- quality gain and cost saving versus the calibration-best fixed model;
- paired dataset-stratified bootstrap intervals;
- full cost-quality curves and regret to hindsight oracle;
- separate ID and OOD results.

This public one-shot study does not validate long-horizon agent routing. The
officially graded Tinker SWE-bench screens and earlier temporal studies remain
the agentic evidence and must be reported separately.
