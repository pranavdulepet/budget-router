# Amendment 009: three-tier official agent quality-cost curve

Status: **prospective freeze before any affected paid inference**

Effective date: 2026-07-30

Parent study: `open-router-study-2026-07-29-v1`

## Authorization and purpose

The user explicitly requested a proper RouteLLM-style quality-versus-cost
curve using genuinely distinct cheap, medium, and strong models, while
retaining official agent correctness. The user subsequently authorized
completion of the study without the prior USD 220 constraint, while asking
that money be used only where it strengthens the evidence.

This amendment supersedes the model-pool, sample-size, and incremental-budget
sections of the parent protocol for the new prospective study. Amendments
001--008 remain completed evidence and design history. Their outcomes are not
reinterpreted as Amendment 009 held-out evidence.

The new incremental hard ceiling is **USD 3,000.00**. It is a safety ceiling,
not a spending target, and is separate from the completed USD 220 program.

## Complete Tinker catalog review and frozen model roles

Tinker's complete live catalog was reviewed on 2026-07-30, including all 16
distinct current model weights, alternate-context variants, base models, and
the retired-model list:
<https://tinker-docs.thinkingmachines.ai/tinker/models/>.

The final pool uses Tinker's own published size categories rather than
project-invented tier names:

| Role | Model | Tinker size | Context | Prefill / sample per 1M | Episode cap |
|---|---|---|---:|---:|---:|
| cheap | `openai/gpt-oss-20b` | Small | 32K | $0.18 / $0.45 | $0.35 |
| medium | `Qwen/Qwen3.6-35B-A3B` | Medium | 64K | $0.54 / $1.335 | $0.90 |
| strong | `Qwen/Qwen3.5-397B-A17B` | Large | 64K | $3.00 / $7.50 | $5.00 |

The price ladder is approximately 3x from cheap to medium and 5.6x from
medium to strong. All are post-trained reasoning or hybrid models with native
agent/tool support. GPT-OSS 20B and Qwen3.6 35B are already same-harness
eligible, resolving 13/60 and 24/60 respectively in Amendment 008.
Qwen3.5 397B is newly admitted because its official model card reports broad
agent evaluation, including SWE-bench Verified and Terminal-Bench 2, and the
current Tinker cookbook provides a dedicated Qwen3.5 tool renderer.

The remaining catalog models are not new study arms. Base models are
ineligible for this instruction-following harness; extended-context variants
are unnecessary; Qwen3-8B previously resolved 1/32; GPT-OSS 120B and
Nemotron Nano had structural failures; and prior same-harness Inkling, Kimi,
DeepSeek, Qwen27B, Nemotron Super, and Nemotron Ultra evidence did not support
a clearer cheap, medium, or strong role. This is a frozen selection, not a
post-hoc screen among many paid challengers.

## Strong-model compatibility gate

Before any new matrix collection:

1. run a two-turn synthetic Qwen3.5 tool-call and tool-result smoke;
2. run fixed Qwen3.5 397B on six previously graded development tasks using
   the locked agent harness;
3. open official grades only after all six records are terminal.

The six tasks are:

- `pydata__xarray-3677`;
- `pytest-dev__pytest-5631`;
- `pytest-dev__pytest-7205`;
- `pytest-dev__pytest-10081`;
- `pydata__xarray-7229`;
- `pydata__xarray-4695`.

Proceed only if the synthetic interaction is structurally valid, at least
five of six agent episodes are structurally valid, at least two tasks resolve
officially, there is no systematic renderer/tool defect, and there are no
unclassified infrastructure errors. Failure stops Amendment 009 paid work.
No result-dependent model replacement is allowed.

## Data split

After the compatibility gate, freeze exact task IDs before matrix inference.

### Development

- 60 SWE-bench Verified tasks;
- 15 each from `astropy/astropy`, `matplotlib/matplotlib`,
  `pydata/xarray`, and `pytest-dev/pytest`;
- deterministic SHA-256 selection seed
  `three-tier-router-development-20260730`;
- no task may appear in the compatibility gate or held-out cohort.

Every model runs on every development task under the same scaffold. Existing
records may be reused only when task, model, renderer, prompt, seed policy,
context treatment, output allowance, step limit, and price accounting are
identical and their artifact identity is recorded. Otherwise the cell is
recollected.

### Held-out

- 100 exact-task-unseen SWE-bench Verified tasks;
- 32 from `django/django`;
- all 14 currently unused tasks from `scikit-learn/scikit-learn`;
- all 26 currently unused tasks from `sphinx-doc/sphinx`;
- 28 from `sympy/sympy`;
- deterministic SHA-256 selection seed
  `three-tier-router-heldout-20260730`.

"Unused" means no prior paid episode, saved trajectory, terminal shard,
submitted prediction, or official project grade for the exact task ID. The
selection script must scan all project artifacts and outputs before freezing
the manifest. Held-out repositories are disjoint from Amendment 009
development repositories. The imbalanced task allocation is reported, and
repository-macro estimates accompany task-micro estimates.

No held-out gold patch, grader test, label, grader log, or model outcome may
enter fitting, calibration, threshold choice, debugging, or policy selection.

## Complete fixed-model matrix

Run all three fixed models on all 60 development tasks and all 100 held-out
tasks. The fixed matrix has three purposes:

1. complete, same-harness cheap/medium/strong baselines;
2. supervised development data for the router;
3. exact official counterfactual evaluation of task-level policies.

The held-out fixed-model episodes all finish before any official held-out
grade is opened. Unsubmitted, malformed, and budget-exhausted episodes remain
unresolved in the denominator.

## Task-level router and RouteLLM-style curve

Fit shared task-model success classifiers on development only. Each row
combines public issue/repository features with provider-neutral model-card
features. Required learned candidates are:

- signed hashed logistic regression with calibrated probabilities;
- calibrated gradient boosting;
- text nearest-neighbor outcome regression.

Required controls are:

- every fixed model;
- repository lookup with global fallback;
- uniform random;
- frequency-matched random;
- cheapest model;
- development-best fixed model;
- hindsight task oracle, labeled non-deployable.

Hyperparameters and calibration use leave-one-development-repository-out
predictions. Each learned method emits the common decision rule

`predicted_success - lambda * normalized_expected_cost`

over a dense, predeclared lambda grid from 0 through 4. The primary task
router is the simplest method on the development Pareto frontier, with ties
broken by higher cross-validated success, lower expected cost, better Brier
score, then lower complexity.

All held-out task-route decisions are frozen from public inputs before grades
are opened. Because every fixed-model trajectory is already collected, a
task-level route selects the corresponding fixed prediction and exact
observed cost. This produces a full official-correctness quality-cost curve
without duplicate inference. Random curves use 10,000 deterministic
task-stratified assignments and seed `20260730`.

## Agent-step router

The agent-step router is a provider-neutral shared classifier. At every model
call it estimates each eligible model's probability of terminal task success
from:

- the original public issue;
- repository and task features;
- the selected model card;
- visible trajectory messages and tool results;
- call, error, test, diff, context, and remaining-budget summaries.

It never reads hidden reasoning, reference patches, official tests, grades,
or future state. Development labels are the official terminal outcomes of
the corresponding fixed-model trajectories, grouped by task to prevent
prefix leakage. Calibration and threshold selection use
leave-one-development-repository-out predictions.

Routing is monotone within an episode: cheap may escalate to medium or strong,
and medium may escalate to strong; the router never de-escalates. This avoids
cache-thrashing and makes the intervention interpretable. Escalation occurs
only at complete visible turn boundaries. Provider failures may fall forward
one tier on the unchanged visible state, with the failed reservation charged.

Freeze three on-policy operating points from development only:

- **cost**: highest cross-validated quality whose expected cost is no greater
  than fixed medium;
- **balanced**: the knee of the development Pareto curve by maximum distance
  from the fixed-endpoint chord after min-max normalization;
- **quality**: cheapest policy whose cross-validated resolved count matches
  the development-best fixed model.

If an exact constraint has no feasible point, use the nearest feasible
development point and record the gate failure; do not inspect held-out labels.
Run the three frozen policies on 12 development tasks each as a structural and
activation check. Each must have at least 11/12 structurally valid episodes,
no provider failures, at least two represented model tiers, and no policy
invariant violation. Quality is not a launch gate for these development
checks.

## Held-out treatments and grading lock

Run six on-policy treatments on every held-out task:

1. fixed cheap;
2. fixed medium;
3. fixed strong;
4. frozen agent-cost;
5. frozen agent-balanced;
6. frozen agent-quality.

Treatment order is deterministically shuffled within task with seed
`20260730`. All 600 canonical episodes must be terminal before official
held-out grading begins. The task-router and random-control curves reuse the
three fixed rows and therefore add no provider calls.

## Analysis and claim rules

Primary quality is official SWE-bench resolved count. Primary cost is exact
conservative uncached-list-price inference cost from observed tokens.

The primary agent comparison is agent-balanced versus the held-out-best fixed
model. A positive headline requires:

- agent-balanced resolves at least as many tasks;
- agent-balanced has lower total cost;
- at least two model tiers are used across held-out trajectories; and
- at least 25/100 trajectories use a tier cheaper than the strongest tier.

Also report:

- all fixed, task-router, agent-router, random, lookup, and oracle points;
- quality at matched cost and cost at matched quality;
- Pareto frontier, dominated points, and normalized frontier area;
- total cost, mean cost, cost per resolution, calls, switches, and route share;
- task-micro and repository-macro estimates;
- paired task and repository bootstrap intervals with 10,000 resamples;
- exact McNemar discordances for named policy comparisons;
- calibration, Brier score, ECE, uncertainty, and abstention/error analysis;
- results with list prices as frozen and a price-sensitivity replay.

The task-level curve, agent-level curve, and external frontier references are
reported separately. Vendor model-card scores are context, not same-harness
results. A failure of the positive rule is an informative result and cannot
be repaired with post-hoc held-out tuning.

## Ordered budget

| Stage | Maximum incremental exposure |
|---|---:|
| synthetic compatibility smoke | $0.50 |
| six-task strong compatibility gate | $30.00 |
| 60-task three-model development matrix | $375.00 |
| 36 on-policy development checks | $180.00 |
| 100-task three-model held-out fixed matrix | $625.00 |
| three 100-task held-out agent policies | $1,500.00 |
| infrastructure/grader recovery and contingency | $289.50 |
| **absolute Amendment 009 total** | **$3,000.00** |

Every call reserves its full episode cap against its stage and the amendment
ceiling. Actual unused authorization remains unspent. No stage may silently
add tasks, models, seeds, or policies. If Tinker credit is insufficient, stop
at the current terminal boundary and ask the user to add only the amount
required for the next complete stage.

## Completion

After grading, update the technical report, blog, public result artifacts,
README, examples, and provider-neutral router API. The release must reproduce
all task-level decisions from the frozen artifact and expose the same
task-model and visible-agent-step interfaces for arbitrary user-supplied
models and executors.
