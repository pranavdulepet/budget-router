# Final report: isolated diagnostic scouting and a learned policy gate

## Outcome

This follow-up found genuine held-out policy complementarity, but the frozen
learned gate did not capture it.

On 39 repository-disjoint SWE-bench Verified tasks:

| Policy | Submitted | Resolved | 95% Wilson interval | Total cost | Cost / resolved |
|---|---:|---:|---:|---:|---:|
| Fixed Qwen 3.6 35B-A3B | 28 | 15/39 (38.5%) | 24.9%–54.1% | $19.6834 | $1.3122 |
| GPT-OSS 20B scout → clean Qwen finisher | 19 | 13/39 (33.3%) | 20.6%–49.0% | $25.1480 | $1.9345 |
| Frozen issue-text gate | 28 | 15/39 (38.5%) | 24.9%–54.1% | $19.6834 | $1.3122 |
| Hindsight policy oracle | — | 19/39 (48.7%) | 33.9%–63.8% | $19.3033 | $1.0160 |

The universal isolated policy lost two resolutions and cost $5.4646 more than
fixed Qwen. The paired resolution-rate delta was -5.13 percentage points, with
a 10,000-sample paired-bootstrap 95% interval of -20.51 to +10.26 points.
There were four isolated-only resolutions and six fixed-only resolutions; the
two-sided exact McNemar p-value was 0.7539.

The learned gate selected fixed Qwen on all 39 tasks and therefore matched the
fixed baseline exactly. This was the correct decision according to its
training data, where the isolated policy had zero unique successes and was
more expensive. It was not the hindsight-optimal decision on the held-out
repositories, where four new isolated-only successes appeared.

The hindsight oracle is not deployable. It uses terminal outcomes to select a
policy. Its 19 resolutions and $19.3033 cost establish an observed upper bound:
perfect policy choice could have added four resolutions while spending $0.3801
less than fixed Qwen.

## Research question

The prior same-workspace Qwen 8B → Qwen 35B cascade was cheaper but lost two
resolutions. This preregistered follow-up tested a cleaner and more interesting
hypothesis:

1. Can a cheap diagnostic model inspect a disposable workspace and transfer
   useful visible evidence to a strong model in a fresh workspace?
2. Can a classifier trained only on public issue text choose when that policy
   is preferable to fixed Qwen?

The study manifest was frozen before any new provider call:

- study:
  `isolated-stage-router-2026-07-28-v1`;
- manifest hash:
  `0ccea14c09ed1f11eebf33df95adb8370f38ab1feb59dc776ffe6089f4e4e48d`;
- maximum frozen exposure:
  $112.50;
- actual incremental exposure:
  $73.142118285.

## Treatments

The fixed comparator was Qwen/Qwen3.6-35B-A3B running the bounded coding agent
under a $0.90 hard cap.

The isolated treatment used:

- openai/gpt-oss-20b as a six-call diagnostic scout;
- a $0.08 scout sub-cap;
- a disposable scout workspace;
- transfer of visible notes, shell commands, tool outputs, and harness
  feedback only;
- removal of hidden reasoning;
- a 60,000-character sanitized handoff limit;
- a fresh implementation workspace for Qwen/Qwen3.6-35B-A3B;
- a finisher cap equal to $0.90 minus actual scout spend;
- the same 75-step total limit as fixed Qwen.

Scout submissions and workspace edits were discarded. This isolates the value
of diagnosis and visible evidence from the value of directly editing the final
workspace.

## Frozen design

The experiment used the pinned SWE-bench Verified revision
`91aa3ed51b709be6457e12d00300a6a596d4c6a3` and official harness commit
`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`. The local dataset snapshot
SHA-256 was
`e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076`.

The design contained:

- a six-task screen for three scouts, 18 episodes total;
- a 29-task expansion for the selected scout;
- 35 gate-training tasks across five repositories;
- 39 held-out paired tasks across six different repositories;
- 78 held-out episodes, one fixed and one isolated per task.

Repositories were disjoint between gate training and the final test. Test
outcomes were not accessed before the candidate and gate artifacts were
cryptographically frozen.

## Candidate screen

The scout candidates were Qwen 3 8B, GPT-OSS 20B, and GPT-OSS 120B, each
followed by the same Qwen finisher. Every candidate resolved 2/6 screen tasks,
and none added a success beyond fixed Qwen.

GPT-OSS 20B was the only candidate satisfying every frozen eligibility rule:
all six episodes were structurally valid, there were no provider failures, and
the official grader had no error. Its candidate freeze hash is
`05bc8f1c6d3e16693bc740fc13ab066adff75ed95f65fdf7becac32af7cdd164`.

The screen cost $10.072174440. The selected GPT-OSS 20B policy then resolved
9/29 expansion tasks for $18.238539555.

## What was trained

The deployable router is a binary policy gate, not an LLM judge. It reads only
the public problem statement. It fits two balanced hashed-logistic heads:

- probability of an isolated-only win;
- probability of a fixed-only loss.

The 1,024-dimensional signed feature hash contains normalized unigrams,
adjacent bigrams, and a coarse text-length feature. Repository identity, gold
patches, grader tests, grader logs, and test outcomes are excluded.

For a task \(x\), the frozen score is

\[
s(x)=\hat p(\text{isolated-only win}\mid x)
     -\hat p(\text{fixed-only loss}\mid x)
     +\lambda r_C,
\]

where \(r_C\) is the training-mean cost saving ratio. The gate chooses the
isolated policy when \(s(x)\) exceeds a threshold.

Hyperparameters, cost weight, and threshold were selected with
leave-one-repository-out cross-validation. The selected settings were learning
rate 0.2, L2 0.001, 200 epochs, cost weight 0, and threshold 0.3.

The training evidence was unambiguous:

| Training outcome | Isolated | Fixed |
|---|---:|---:|
| Resolved / 35 | 11 | 16 |
| Unique wins | 0 | 5 |
| Total cost | $21.5427 | $17.9206 |

Cross-validation selected fixed on all 35 tasks and resolved 16. The exact
gate artifact hash is
`ecfcf934be32222df01d27072b6e298b88323c4706743392cc3fdc25c7bbfb86`;
the pre-test gate freeze hash is
`53bf4d83804b33f6211389af1771c06d00c17d9f6090c0b10d00d7669ab13dc3`.

## Official held-out evaluation

All 39 paired episodes completed. Forty-seven patches were submitted and
graded; 31 no-patch episodes remained unresolved in the denominator.

Two fixed-Qwen patches were malformed and could not be applied:
`pydata__xarray-3151` and `sphinx-doc__sphinx-8475`. Both failures reproduced
on retry and are classified as unresolved model outcomes. A separate Docker
pull timeout on `pytest-dev__pytest-7324` succeeded on retry and its patch
received an ordinary unresolved grade. No unclassified grader or
infrastructure errors remain.

The four isolated-only resolutions were:

- `pydata__xarray-3151`;
- `pydata__xarray-6721`;
- `pytest-dev__pytest-7324`;
- `sphinx-doc__sphinx-8475`.

The six fixed-only resolutions were:

- `pytest-dev__pytest-8399`;
- `scikit-learn__scikit-learn-25973`;
- `sphinx-doc__sphinx-8721`;
- `sphinx-doc__sphinx-9658`;
- `sympy__sympy-15345`;
- `sympy__sympy-15875`.

This is real complementarity: the observed policy oracle resolved 19 tasks,
four more than either deployable choice. It also exposes a generalization
failure. Gate training contained no isolated-only win, while the held-out
repositories contained four.

## Did the cheap scout save money?

Not as a universal policy.

Across 39 isolated episodes:

- GPT-OSS made 234 calls and cost $0.226447560;
- Qwen finishers made 2,113 calls and cost $24.921575025;
- the scout was only 0.90% of isolated-policy cost;
- the isolated policy cost less on 13/39 tasks;
- it cost more on 26/39 tasks;
- it was cheaper with an equal-or-better outcome on 13/39 tasks;
- it was cheaper with both policies resolving on 3/39 tasks.

The cheap model itself was inexpensive. The problem was downstream: a fresh
Qwen finisher often used enough work to make total treatment cost exceed fixed
Qwen. Cheap scouting is therefore not automatically cost-saving.

## Interpretation

Three claims are supported:

1. The universal isolated policy did not beat fixed Qwen. It resolved fewer
   tasks and cost more.
2. The frozen trained gate safely fell back to fixed Qwen and matched it.
3. The observed test outcomes contain meaningful policy complementarity that
   the training distribution did not reveal.

The third finding matters. This is not evidence that routing has no value.
It is evidence that a task-only classifier cannot learn a boundary absent from
its supervision, especially across a repository shift.

The result suggests that the next router should use online evidence rather
than issue text alone: scout diagnostics, test failures, touched modules,
uncertainty, or a structured checkpoint. Those features exist before terminal
outcomes and could distinguish the four isolated-only cases from the six
fixed-only cases. They must be trained on new paired data and evaluated on a
fresh repository-disjoint test.

The present 39 labels cannot be used both to train that richer gate and to
claim held-out performance. They can support exploratory feature design and
power analysis, not a new confirmatory score on the same cohort.

## Cost and scope

Recorded exposure was:

- screen: $10.072174440;
- expansion: $18.238539555;
- held-out paired test: $44.831404290;
- total: $73.142118285.

That is $39.357881715 below the frozen $112.50 ceiling. No additional Tinker
spend is required to complete this result.

This is a custom, budgeted, single-seed treatment comparison. It is not an
official public SWE-bench leaderboard submission or a universal model ranking.
Uncertainty remains wide, and results are specific to the frozen agent,
budgets, prompts, models, prices, and repository split.

## Reproduction and artifacts

The sanitized public result is
[`artifacts/isolated_stage_v1_results.json`](../artifacts/isolated_stage_v1_results.json).
The exact pre-test classifier is
[`artifacts/isolated_stage_v1_task_gate.json`](../artifacts/isolated_stage_v1_task_gate.json).
The protocol is
[`artifacts/isolated_stage_router_study_v1.json`](../artifacts/isolated_stage_router_study_v1.json).

The result artifact pins the dataset, harness, reports, grades, evaluation,
gate, freezes, costs, intervals, and task-level policy selections. Raw paid
trajectories, hidden reasoning, patches, terminal logs, and credentials are
not part of the public artifact.

After official grading is complete, reproduce every derived result with:

```bash
python scripts/finalize_isolated_stage_results.py
```

The public result hash is
`61e97c90514c7096f7c8fbaa83ef27a96bf42d72743ff3c80aeb59ab44191209`.
