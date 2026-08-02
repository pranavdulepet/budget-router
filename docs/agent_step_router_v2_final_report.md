# Guarded agent-step routing on SWE-bench

## Executive summary

We trained and prospectively evaluated a classifier that chooses GPT-OSS 20B
or Qwen3.6 35B-A3B before every LLM call inside a coding-agent trajectory.
The router passed its frozen joint criterion on 60 previously unexecuted
SWE-bench Verified tasks:

| Policy | Submitted | Officially resolved | Resolution rate | Cost |
|---|---:|---:|---:|---:|
| Fixed GPT-OSS 20B | 40/60 | 13/60 | 21.67% | $9.5290 |
| Fixed Qwen3.6 35B | 42/60 | 24/60 | 40.00% | $30.9780 |
| Guarded per-call router | 44/60 | 26/60 | 43.33% | $24.5376 |

Relative to fixed Qwen, the router resolved two more tasks and saved
$6.4404, or 20.79%. Five tasks were router-only successes and three were
Qwen-only successes. Exact McNemar \(p=0.7266\); the paired-bootstrap 95%
interval for the quality difference was [-6.67, +13.33] percentage points.
The data therefore do not prove a quality improvement. The cost-saving
interval was [9.91%, 30.57%], which supports a real policy-level cost
reduction on this cohort.

Unlike the earlier 20-task experiment, the mechanism activated. The router
sent 364/2,866 calls (12.70%) to GPT-OSS and used GPT-OSS at least once on
56/60 trajectories. It passed every frozen activation and guard requirement.

The predeclared primary transfer subgroup contained 50 tasks from repository
families excluded from classifier fitting and live threshold development.
There, the router and Qwen tied at 22/50 while the router saved 20.80%. Its
paired quality interval was [-10, +10] points and its cost-saving interval was
[11.34%, 30.13%].

This is positive evidence for the intended routing paradigm: a small,
inspectable classifier made real cheap-model substitutions and matched the
strong fixed baseline's observed quality at lower cost. It is not proof of
universal non-inferiority, a statistically significant quality gain, or
transfer to arbitrary agents and model pools.

## What was tested

The research question was:

> Can a guarded classifier choose a cheaper model for selected calls inside a
> coding agent while matching or exceeding a fixed-strong baseline's official
> task resolution at lower total cost?

Amendment 008 was frozen before the new paid episodes. It specified:

- GPT-OSS 20B as the cheap tier and Qwen3.6 35B-A3B as the strong tier;
- public execution-verified step supervision;
- repository exclusions, grouped splits, model fitting, and calibration;
- a live-prefix threshold rule that did not use terminal task outcomes;
- initial-strong, context, dwell, and maximum-cheap-burst guards;
- a 20-task development gate;
- 60 new tasks and three independently sampled treatments per task;
- a 50-task novel-repository primary subgroup;
- an official pinned SWE-bench grader;
- paired task and repository analyses;
- hard episode and cumulative dollar caps.

The two models were not chosen arbitrarily. Earlier frozen screens tested
GPT-OSS 20B, GPT-OSS 120B, Nemotron Nano, DeepSeek V3.1, Qwen3.6 27B, and
Nemotron Ultra against the Qwen3.6 35B control. GPT-OSS 20B was the only
eligible cheap-screen winner; the higher-cost candidates produced no
official screen resolutions, and one candidate was infrastructure
incompatible. Qwen3.6 35B was the established strong coding-agent baseline.
The selected pair offered both an observed quality gap and roughly threefold
list-price separation.

## Router design

Before each model call, the router sees only the bounded visible prefix:
system and user messages, prior assistant outputs, tool calls, and tool
results. It does not see gold patches, grader tests, hidden reasoning, future
messages, or task outcomes.

The frozen artifact contains:

- a 16,384-dimensional signed hashed-logistic classifier;
- Platt probability calibration;
- a 0.473756501959466 cheap threshold;
- opaque cheap and strong model IDs;
- a 24,000-token strong-model context guard;
- two mandatory strong calls at the beginning of every trajectory;
- at most two consecutive cheap calls;
- a two-call strong dwell before de-escalation;
- no credential or provider client.

The classifier was refit on 459 public training rows and calibrated on 112
rows after excluding the new test repository families. The threshold was the
lowest observed score that satisfied the live activation constraints on
1,129 already-visible prefixes from the prior 20-task experiment. Their
official terminal outcomes were not used.

This is a stateful policy, not a one-time easy/medium/hard label. A strong
decision can force a dwell; long context forces Qwen; a cheap burst is
bounded; and every new trajectory begins strong.

## Development gates

Shadow replay on the 20 visible development trajectories predicted:

- 170/1,129 cheap calls, or 15.06%;
- cheap use on 18/20 trajectories;
- no initial-guard or cheap-burst violations.

The unchanged artifact then ran actively on those 20 development tasks. It
resolved 11/20 under the official grader, used 121/1,081 cheap calls
(11.19%), used GPT-OSS on 16/20 trajectories, produced 20/20 structurally
valid episodes, and had no provider or unclassified grader errors. That
passed every frozen gate and authorized the new held-out cohort. No parameter
changed afterward.

## Held-out design

The blind test contained 60 previously unexecuted task IDs. Each received
three independent treatments:

1. fixed GPT-OSS 20B;
2. fixed Qwen3.6 35B-A3B;
3. the frozen guarded router.

The treatment order was deterministically blocked by task. Every arm used the
same mini-SWE-agent scaffold, task ordinal, tool interface, maximum 75 calls,
and policy-specific hard cap. All 180 canonical episodes were terminal before
any official held-out grade was opened.

The primary transfer subgroup had 50 tasks:

- 27 Django;
- 18 Matplotlib;
- 5 Astropy.

Those repository families were excluded from classifier training,
calibration, and threshold development. A secondary 10-task subgroup used two
tasks each from xarray, pytest, scikit-learn, Sphinx, and SymPy.

The allocation is intentionally disclosed as imbalanced. Repository-macro
estimates accompany task-micro results so Django cannot silently dominate
every interpretation.

## Execution and grading integrity

All 180 episodes were terminal and provider-failure-free. The held-out
treatments cost $65.044578375:

- fixed GPT-OSS: $9.529009650;
- fixed Qwen: $30.977961915;
- guarded router: $24.537606810.

Development plus held-out exposure was $74.576625660 against the frozen
$150 maximum. Unused reserve was not spent merely because it was available.

Predictions were graded with the official SWE-bench terminal harness at
commit `f7bbbb2ccdf479001d6467c9e34af59e44a840f9` against dataset SHA-256
`e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076`.

One fixed-Qwen diff for `django__django-11749` used an absolute
`/testbed/...backup` header and could not apply. A repeat produced the same
failure, confirming model-caused malformed output. It is preserved and
counted unresolved.

Two router images initially failed to download because Docker Hub timed out
before evaluation. The resume-safe harness skipped all 42 completed router
reports and reran only those two tasks. One resolved and one completed
unresolved. The final grading manifest contains no router or infrastructure
errors and one explicitly classified Qwen model-output error.

## Primary results

### Full 60-task operational comparison

Router versus fixed Qwen:

- resolutions: 26 versus 24;
- difference: +2 tasks, or +3.33 percentage points;
- router-only/Qwen-only: 5/3;
- exact McNemar \(p=0.7265625\);
- paired quality-difference 95% interval: [-6.67, +13.33] points;
- total cost: $24.5376 versus $30.9780;
- saving: $6.4404, or 20.79%;
- paired cost-saving 95% interval: [9.91%, 30.57%].

The frozen joint endpoint passed all four components:

1. router resolution at least fixed-Qwen resolution;
2. router cost below fixed-Qwen cost;
3. router cheap-call share at least 10%;
4. at least 30 routed trajectories containing a cheap call.

The quality point estimate favors the router but the interval includes
meaningful losses and gains. The defensible claim is observed quality
compatible with fixed Qwen at materially lower cost, not that the router is
proven better.

### Novel-repository primary transfer analysis

| Policy | Resolved | Cost |
|---|---:|---:|
| Fixed GPT-OSS | 12/50 | $8.1083 |
| Fixed Qwen | 22/50 | $26.0217 |
| Guarded router | 22/50 | $20.6087 |

Router versus Qwen:

- quality difference: 0 points;
- discordance: 3 router-only and 3 Qwen-only;
- exact McNemar \(p=1.0\);
- paired quality interval: [-10, +10] points;
- cost saving: 20.80%;
- paired cost-saving interval: [11.34%, 30.13%].

This is the cleanest evidence in the study. On repository families absent
from fitting and threshold development, the router tied the strong model and
saved about one-fifth of cost.

Equal-weighting Astropy, Django, and Matplotlib changes the quality point
estimate to -6.05 points because the task allocation is imbalanced:

| Repository | Tasks | GPT-OSS | Qwen | Router |
|---|---:|---:|---:|---:|
| Astropy | 5 | 0 | 1 | 0 |
| Django | 27 | 8 | 13 | 15 |
| Matplotlib | 18 | 4 | 8 | 7 |

The novel-repository macro quality interval was [-20.0, +7.41] points. Its
macro cost saving was 16.49%, with an [8.59%, 25.85%] interval. Only three
repository clusters exist, so this macro result is especially uncertain.

### Familiar-repository secondary analysis

On the 10 familiar-repository tasks, the router resolved 4, Qwen 2, and
GPT-OSS 1. Router-only/Qwen-only discordance was 2/0, exact McNemar
\(p=0.5\), and the quality interval was [0, +50] points. The router saved
20.73% in the point estimate, but the cost interval [-39.32%, +52.32%] was
extremely wide. This subgroup is too small for a strong standalone claim.

## Mechanism audit

| Measure | Observed |
|---|---:|
| Router model calls | 2,866 |
| GPT-OSS calls | 364 |
| Qwen calls | 2,502 |
| Cheap-call share | 12.70% |
| Trajectories with a cheap call | 56/60 |
| Model switches | 400 |
| Guard-compliant trajectories | 60/60 |

Forced-strong activations were:

- 120 initial calls;
- 389 context guards;
- 72 dwell guards;
- 127 cheap-burst guards.

The GPT-OSS calls consumed 5,140,536 input and 112,429 output tokens and cost
$0.9759. Repricing those exact tokens at Qwen's list rates gives $2.9260, a
$1.9501 mechanical substitution saving. This same-token calculation isolates
price. It is not the full causal policy effect because changing a model also
changes later commands, tokens, call counts, and whether the agent stops.

The randomized policy-level comparison captures the total observed effect:
the router used 2,866 calls versus fixed Qwen's 3,226 and saved $6.4404.
Part of that difference is direct model price and part is changed trajectory
behavior.

Shadow replay on the independently sampled fixed-Qwen prefixes selected
GPT-OSS for 502/3,226 calls (15.56%) and on 55/60 trajectories. Only five
fixed-Qwen trajectories would remain all-Qwen. This identity audit confirms
that the classifier's activation was not peculiar to the actively routed
prefixes, while also showing that active routing changes the states it later
observes.

## Baselines and oracle ceiling

GPT-OSS established the low-cost/low-quality endpoint: 13 resolutions at
$9.5290, or $0.733 per resolution. Qwen reached 24 at $30.9780, or $1.291
per resolution. The router reached 26 at $24.5376, or $0.944 per resolution.
It therefore improved observed Qwen quality while reducing both total cost
and cost per resolved task.

A non-deployable oracle choosing retrospectively between the two fixed
models resolved 27/60. An oracle allowed to choose any of all three sampled
arms resolved 31/60. These use hidden terminal outcomes and independently
sampled realized costs, so they are opportunity diagnostics, not deployable
policies or fair production cost estimates.

The fixed-model oracle's 27 versus Qwen's 24 shows genuine task-level
complementarity. The learned router's 26 is close to that observed ceiling,
but it does not mean the router identified the same tasks or exhausted the
model pool's potential.

## What is scientifically supported

The study supports these conclusions:

1. A real classifier can route individual agent calls, not only whole
   prompts or tasks.
2. Representative live-prefix calibration fixed the earlier zero-activation
   failure without using held-out task outcomes.
3. On 50 novel-repository tasks, the frozen router matched fixed Qwen's
   observed resolution and saved about 21%.
4. On all 60 tasks, it resolved 26 versus 24 while saving about 21%, passing
   the predeclared joint endpoint.
5. The quality difference remains statistically uncertain; a small loss or
   a moderate gain is compatible with the data.
6. Direct cheap-token repricing explains $1.95 of mechanical savings, while
   the full policy changed trajectory length and behavior.
7. Repository-level transfer is heterogeneous: the router led on Django,
   trailed on Astropy and Matplotlib combined, and the macro interval is wide.

The study does not establish universal model rankings, a five-point
non-inferiority margin, calibration on unrelated agents, robustness after
model or price updates, or a benefit from adding arbitrary extra models.

## Open-source use

`GuardedAgentStepArtifact` and `GuardedAgentStepRouter` load the exact frozen
two-tier artifact and select a model from visible messages before each call.
`SemanticAgentRouter` adapts an arbitrary complete prompt- or prefix-by-model
outcome matrix to any supplied model cards, prices, providers, capabilities,
and compatible endpoints.

The library returns a decision and audit metadata; the caller owns provider
execution, agent state, retries, evaluation, and feedback. The router artifact
contains no key. See
[`docs/open_source_router.md`](open_source_router.md) for CLI, Python, proxy,
and arbitrary-pool training examples.

## Remaining research

This 60-task cohort must not be retuned and reported again. Useful next work
would be:

1. repeat the unchanged artifact on a new time- or repository-disjoint
   benchmark to narrow the quality interval;
2. add more cheap/medium candidates only after collecting complete
   prefix-by-model outcomes and passing agent-protocol screens;
3. learn selective abstention for unsupported prefixes and repository drift;
4. compare a task-level router, a per-call router, and a combined hierarchy
   under the same model pool and budgets;
5. add online latency/provider-health routing beneath semantic model choice;
6. use multiple trajectory seeds per task when budget permits, separating
   task difficulty from sampling variance.

The current result is sufficient for the blog's core claim: under a frozen,
leakage-controlled design, agent-level classifier routing matched the strong
baseline's observed quality at materially lower cost. Larger external
replications would strengthen generality, not repair a failed mechanism.

## Reproducibility

- Protocol amendment:
  [`artifacts/active_router_protocol_amendment_008_guarded_agent_step_followup.json`](../artifacts/active_router_protocol_amendment_008_guarded_agent_step_followup.json)
- Task manifest:
  [`artifacts/agent_step_followup_v2_task_manifest.json`](../artifacts/agent_step_followup_v2_task_manifest.json)
- Sanitized result:
  [`artifacts/agent_step_router_v2_results.json`](../artifacts/agent_step_router_v2_results.json)
- Grading recovery audit:
  [`artifacts/agent_step_router_v2_grading_recovery.json`](../artifacts/agent_step_router_v2_grading_recovery.json)
- Router artifact:
  `outputs/agent_step_router_v2/router_artifact.json`
- Official grading manifest:
  `outputs/agent_step_router_v2/swebench_grader/heldout/grading_manifest.json`
- Analysis script:
  `scripts/analyze_guarded_agent_step_followup.py`

The router artifact hash is
`aca70d306b11ce33b3d2c762fbcaf3f7d642a8b1711643b54363b7df90f3ca14`.
The task manifest hash is
`b213b768a3e00a1cf286a77819da044d267544c75ee75823b0859dc9856f439a`.
The protocol hash is
`d30115410c67d9c25216f5e1de79c0fd78177a2581674d35d31f94b67da151f2`.
