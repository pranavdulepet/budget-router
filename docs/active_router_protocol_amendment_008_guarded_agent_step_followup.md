# Amendment 008: guarded live-calibrated agent-step follow-up

Status: **prospective freeze before new classifier fit or paid inference**

Effective date: 2026-07-29

Parent study: `open-router-study-2026-07-29-v1`

## Why this follow-up exists

Amendment 007 completed a valid 20-task confirmatory experiment. Its frozen
router met the arithmetic quality-cost endpoint but selected Qwen on all 1,129
held-out calls. The intended cheap-model substitution mechanism did not
activate.

The user explicitly requested completion of the remaining agent-routing
experiments with enough evaluation cases and carefully chosen models. This
amendment treats every Amendment 007 trajectory and official outcome as
development evidence. It does not modify or reinterpret that completed
holdout. All Amendment 008 test tasks are exact-task unseen before this freeze.

This is a prospective, result-informed follow-up rather than an independent
replication.

## Model pool

The pool remains:

- cheap: `openai/gpt-oss-20b`;
- strong: `Qwen/Qwen3.6-35B-A3B`.

This is an evidence-based choice, not a convenience choice. GPT-OSS 20B is the
only screened cheap treatment that passed the structural gate and produced
multiple official resolutions. Qwen3.6 35B is the strongest reliable
same-harness control. GPT-OSS 120B failed the structural gate. DeepSeek V3.1,
Qwen3.6 27B, and Nemotron Ultra resolved zero official screen tasks. Nemotron
Nano had a systematic harness incompatibility. Adding those treatments to a
large matrix would spend budget without a supported routing role.

Renderer, sampling, price, context, prompt, and per-episode cap settings remain
identical to Amendment 007.

## Public fit and leakage control

Refit the same 16,384-dimensional binary hashed-logistic classifier and Platt
calibrator from the pinned TwinRouterBench question bank. Use the same
training-only grouped cross-validation and 70/15/15 instance-grouped split,
with a new deterministic seed
`agent-step-router-live-followup-20260729-v2`.

Exclude TwinRouterBench SWE-bench rows from every repository prefix appearing
in the new 60-task test:

- `astropy`;
- `django`;
- `matplotlib`;
- `pydata`;
- `pytest-dev`;
- `scikit-learn`;
- `sphinx-doc`;
- `sympy`.

No Amendment 008 test task, grade, patch, test definition, or grader log may
influence fitting, calibration, threshold selection, or the development gate.

## Live-prefix threshold calibration

Use the 20 completed Amendment 007 routed trajectories as development-only
shadow prefixes. Their official terminal outcomes do not enter threshold
selection.

For every candidate threshold equal to an observed calibrated probability,
replay the complete visible prefixes with the frozen guarded state machine.
Select the lowest threshold satisfying all of:

- shadow cheap-call share at least 15% and at most 20%;
- at least 15 of 20 development trajectories contain a cheap call;
- the first two calls of every trajectory are forced strong;
- at most two consecutive cheap calls;
- at least two strong calls after entering the strong tier before
  de-escalation;
- the 24,000-token context guard remains active.

The lowest qualifying threshold is the deterministic safety tie-break. The
artifact, training split, source hashes, development-trajectory hashes,
threshold report, and policy guards are serialized and hashed before new paid
inference.

## Paid development activation gate

Run the frozen router once on each of the same 20 Amendment 007 task IDs as an
explicit development stage. Those task outcomes are already open and cannot
enter the new test.

Proceed to the 60-task test only if:

- actual cheap-call share is between 10% and 20%;
- at least 15 of 20 trajectories use the cheap model;
- at least 19 of 20 episodes are structurally valid;
- at least 6 of 20 tasks resolve officially;
- there are no provider failures or unclassified grader errors; and
- every observed trajectory respects both initial-strong and cheap-burst
  guards.

Failure stops the new held-out matrix. No result-contingent threshold change
is permitted within Amendment 008.

## New held-out cohort

Freeze 60 exact-task-unseen SWE-bench Verified tasks before any Amendment 008
provider call.

Fifty form a repository-transfer primary subgroup:

- 27 unused `django/django` tasks;
- 18 unused `matplotlib/matplotlib` tasks;
- 5 unused `astropy/astropy` tasks.

Ten form a familiar-repository exact-task subgroup, two each from:

- `pydata/xarray`;
- `pytest-dev/pytest`;
- `scikit-learn/scikit-learn`;
- `sphinx-doc/sphinx`;
- `sympy/sympy`.

Within each repository, select unused IDs by SHA-256 order with seed
`agent-step-router-v2-heldout-20260729`. “Unused” means no prior paid episode,
saved paid trajectory, terminal episode shard, or official submitted
prediction exists for that exact ID.

The imbalanced novel-repository allocation reflects the finite unused
SWE-bench Verified pool and must be reported. Results include full-task,
repository-macro, 50-task novel-repository, and 10-task familiar-repository
views.

## Treatments and identity audit

Run three independently sampled treatments on all 60 tasks:

1. fixed GPT-OSS 20B;
2. fixed Qwen3.6 35B;
3. the frozen guarded per-call router.

Treatment ordering is blocked by task and deterministically shuffled with
seed `20260729`. Every arm uses the same scaffold, task ordinal, model-specific
seed construction, Docker image, maximum 75 calls, and hard cap.

In addition, replay the frozen router in shadow mode on each fixed-Qwen
trajectory. This zero-cost identity audit reports whether the router would
have selected only Qwen on that exact prefix sequence. It is descriptive and
does not replace the active routed arm.

No official held-out grade is opened until all 180 canonical episodes are
terminal.

## Analysis and claim rule

The primary operational comparison uses all 60 tasks. Joint success requires:

- router resolved count at least fixed-Qwen resolved count;
- router total conservative cost below fixed-Qwen cost;
- actual router cheap-call share at least 10%; and
- at least 30 of 60 routed trajectories contain a cheap call.

The 50-task novel-repository subgroup is the primary transfer analysis. The
10-task familiar-repository subgroup is secondary.

Report:

- every fixed and routed resolution count, submission count, and cost;
- task-, repository-, and subgroup-level route shares;
- cheap and strong calls, switches, guard activations, and trajectory
  positions;
- paired task bootstrap intervals with 10,000 resamples and seed `20260729`;
- exact McNemar discordant counts and p-value;
- repository-macro bootstrap results;
- cost per resolution and cost attributable to actual cheap substitutions;
- model-caused malformed patches separately from infrastructure errors;
- the non-deployable task oracle and shadow identity audit.

A positive routing headline is allowed only if the full joint criterion is
met. Resolution or cost differences between two all-strong executions are not
attributed to routing.

Sixty tasks are three times the prior holdout and the largest complete
three-treatment design that preserves hard-cap recovery room under the
original authorization. It still cannot prove a narrow five-point
non-inferiority margin; uncertainty intervals remain mandatory.

## Budget

Conservative active-router exposure before Amendment 008 is
`$66.739823475`:

- prior active-router exposure recorded by Amendment 007:
  `$43.277878155`;
- Amendment 007 smoke and held-out exposure:
  `$23.461945320`.

Amendment 008 reserves:

| Stage | Maximum |
|---|---:|
| 20 routed development episodes | $18.00 |
| 60 fixed GPT-OSS episodes | $21.00 |
| 60 fixed Qwen episodes | $54.00 |
| 60 routed held-out episodes | $54.00 |
| classified infrastructure recovery reserve | $3.00 |
| **maximum new exposure** | **$150.00** |

The maximum cumulative active-router exposure is `$216.739823475`, below the
original `$220.00` hard ceiling. Unused reserves are not a spending target.
No call may begin if its complete cap would cross the stage, amendment, or
original ceiling.
