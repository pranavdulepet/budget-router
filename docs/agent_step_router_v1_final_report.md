# Agent-step classifier routing on SWE-bench

## Executive summary

We built and held-out tested a classifier that selects a model before every
LLM call in a coding-agent trajectory. The candidate pool was GPT-OSS 20B as
the cheap tier and Qwen3.6 35B-A3B as the strong tier. The comparator arms ran
each model alone. All 60 held-out episodes were collected before official
SWE-bench grades were opened.

| Policy | Submitted | Officially resolved | Resolution rate | Conservative cost |
|---|---:|---:|---:|---:|
| Fixed GPT-OSS 20B | 10/20 | 6/20 | 30% | $3.5905 |
| Fixed Qwen3.6 35B | 11/20 | 8/20 | 40% | $10.1105 |
| Frozen step router | 13/20 | 9/20 | 45% | $9.5095 |

The preregistered arithmetic criterion was met: the router resolved at least
as many tasks as fixed Qwen and its observed total cost was lower. The paired
quality difference was +5 percentage points, with a 10,000-sample bootstrap
95% interval of [-15, +25] points. There were three router-only and two
fixed-Qwen-only resolutions; exact McNemar \(p=1.0\).

The mechanism audit changes the interpretation. Across 1,129 routed calls,
the classifier selected Qwen 1,129 times and GPT-OSS zero times. It never
switched models. The $0.601 observed cost difference therefore cannot be
attributed to cheap-model substitution. It came from stochastic
trajectory/token variation between two all-Qwen executions.

This is neither proof that step routing works nor proof that it cannot work.
It is a clean finding that this frozen classifier and conservative threshold
did not transfer from its public step-label distribution to live
mini-SWE-agent prefixes.

## Research question

The frozen question was:

> Can a classifier selecting a model at every agent LLM call match or exceed
> fixed-Qwen held-out SWE-bench resolution at lower conservative observed
> cost?

The protocol was frozen in Amendment 007 before classifier fitting or new paid
inference. It specified the model pool, public training revision, feature
space, calibration method, threshold rule, static launch gate, 20-task
held-out cohort, three treatments, official grader, cost caps, and analysis.

## Router

Before each model call, the router receives only the visible agent prefix:
system and user messages, prior assistant messages, tool calls, and tool
outputs. It does not receive gold patches, grader tests, hidden reasoning, or
future outcomes.

The deployable artifact contains:

- a 16,384-dimensional signed hashed-logistic classifier;
- Platt probability calibration;
- a threshold that selects GPT-OSS only when the predicted probability of a
  strong-tier requirement is below 0.007158789;
- a 24,000-token strong-model context guard;
- a two-call strong-model dwell before de-escalation;
- opaque cheap and strong model IDs, with no provider credentials.

Training used the Apache-2.0 TwinRouterBench public question bank at revision
`430acecac71141de77afd8e5e13690d236d58e93`. Rows from the held-out
SWE-bench repository families were excluded. Splitting was grouped by
instance, producing 572 training rows, 124 calibration rows, and 158 sealed
static-test rows.

Regularization strength was chosen by five-fold instance-grouped
training-only log loss. Platt calibration and the routing threshold were
chosen on the calibration split. The threshold maximized cheap usage subject
to 100% calibration strong recall and trajectory pass.

## Static launch gate

The frozen artifact passed every prespecified static gate on the untouched
public split:

| Check | Requirement | Observed |
|---|---:|---:|
| Strong-tier recall | at least 95% | 36/37 = 97.30% |
| Cheap call share | at least 10% | 25/158 = 15.82% |
| Trajectory pass | at least 90% | 80/81 = 98.77% |

This justified the dynamic test. It did not guarantee the same route share on
a different agent, prompt format, repository cohort, or trajectory
distribution.

The router artifact hash is
`a6d0468edee1d8f06c74f606d7c1a86b6c5594246832e69c3ef13464f17ec0f8`.
The dynamic-lock manifest hash is
`bd7ca5a8b69392974542c1629f6395e4bc94efd84d170584819c13d0aacdf289`.

## Held-out design

The dynamic test used 20 frozen SWE-bench Verified tasks. Each task received
three independently sampled treatments:

1. fixed GPT-OSS 20B;
2. fixed Qwen3.6 35B-A3B;
3. the frozen per-call router.

The design was blocked by task with frozen treatment ordering. Each episode
had at most 75 model calls. GPT-OSS had a $0.35 cap; Qwen and the router had
$0.90 caps. No held-out official grade was opened until all 60 canonical
episodes were durably recorded.

No provider call failed. Fifty-nine of 60 episodes were structurally valid.
The one invalid GPT-OSS episode is retained as an unresolved treatment
outcome.

## Official grading

Submitted patches were graded with the pinned official SWE-bench terminal
harness at commit `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`
against the pinned dataset snapshot with SHA-256
`e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076`.
Episodes without a submitted patch count unresolved in the 20-task
denominator.

Two submitted outputs could not be applied:

- fixed Qwen produced a diff from `/dev/null` to standard output for
  `sympy__sympy-13372`;
- the router episode for `scikit-learn__scikit-learn-13439` submitted the
  sentence “No patch.txt found, creating from git diff...” instead of a diff.

Their saved predictions and grader logs confirm model-output failures, not
transient Docker or test failures. Both are classified in the grading
manifest and counted unresolved. There are no unclassified official errors.

## Results

### Fixed baselines

GPT-OSS was materially cheaper and weaker: 6/20 resolutions for $3.5905,
versus Qwen's 8/20 for $10.1105. This is the quality-cost separation a router
could potentially exploit. It does not by itself identify which individual
calls are safe for GPT-OSS.

### Frozen router

The raw router result was 9/20 for $9.5095. Relative to fixed Qwen:

- resolution difference: +1 task, or +5 percentage points;
- cost difference: -$0.600983865, or -5.94%;
- paired discordance: three router-only and two fixed-only successes;
- exact McNemar \(p=1.0\);
- paired-bootstrap quality interval: [-15, +25] percentage points;
- paired-bootstrap cost-saving interval: [0.0008%, 12.21%].

The small sample does not establish a quality improvement. More importantly,
the router used no cheap calls:

| Mechanism measure | Observed |
|---|---:|
| GPT-OSS calls | 0 |
| Qwen calls | 1,129 |
| Cheap call share | 0% |
| Model switches | 0 |

The preregistered joint endpoint is mechanically true, but the intended
economic mechanism did not activate. Because the fixed-Qwen and router arms
used independent sampled trajectories, two all-Qwen policies can differ in
tokens, cost, patch submission, and resolution. The result cannot support the
claim that selecting cheaper models caused the cost reduction.

## Why the static and live results diverged

The most likely explanation is distribution transfer. The public training
rows and static test contain a broad mixture of agent-like states and
execution-verified tier labels. Live mini-SWE-agent prefixes have different
system prompts, tool syntax, repository context, message lengths, and state
transitions. Under that shift, every calibrated score exceeded the extremely
conservative cheap threshold.

The threshold also optimized safety before route share. Requiring perfect
calibration strong recall pushed the boundary to 0.0072. That was reasonable
under the frozen rule, but it left no operational margin when live scores
shifted upward.

This exposes a missing launch condition: a static cheap-share gate is not a
dynamic activation gate. A future protocol should require a minimum cheap
share on representative shadow trajectories before paying for a held-out
task-level experiment.

## What is scientifically supported

The study supports four conclusions:

1. A real per-call classifier, calibration layer, stateful dwell rule, and
   provider-neutral runtime were implemented and frozen before test.
2. GPT-OSS and Qwen define a meaningful fixed-policy cost-quality tradeoff on
   this 20-task cohort.
3. The frozen router transferred safely in the narrow sense that it defaulted
   to the strong model, but it failed to make any economical substitutions.
4. Nine versus eight resolutions is sampling-compatible, not evidence that
   the all-Qwen router treatment is intrinsically better than fixed Qwen.

The study does not establish that step routing beats the best fixed model,
that the classifier probabilities remain calibrated on live SWE-bench
prefixes, or that arbitrary model pools will benefit.

## Open-source use

The runtime is not tied to Tinker. `FrozenAgentStepRouter` loads the exact
two-tier experiment artifact and selects a model from visible messages before
every call. `SemanticAgentRouter` adapts an arbitrary-pool semantic artifact
to the same per-call interface and supports model, provider, capability,
context, and per-call-cost constraints.

The router returns a decision; the caller owns provider invocation, agent
state, retries, and terminal evaluation. See
[`docs/open_source_router.md`](open_source_router.md) for CLI and Python
examples.

## Next experiment

The current holdout must remain untouched. A defensible follow-up would:

1. treat these live prefixes as development-only distribution evidence;
2. choose a new threshold on representative shadow trajectories, before
   their terminal labels are used for evaluation;
3. preregister a dynamic activation gate, such as 10–20% cheap calls with a
   strong fallback;
4. freeze a new repository-disjoint task cohort;
5. use an identity control so an all-strong router and fixed-strong treatment
   share the same random seeds and become exactly equivalent;
6. evaluate enough tasks to narrow the current [-15, +25]-point quality
   interval;
7. attribute savings only to calls actually shifted to a cheaper model.

That would test the interesting hypothesis directly. Retuning the threshold
and reporting on these same 20 official outcomes would be leakage.

## Cost and artifacts

The development smoke cost $0.251347305. The 60 held-out episodes cost
$23.210598015, for total new conservative exposure of $23.461945320 against
the frozen $43.90 maximum.

Public, sanitized result:
[`artifacts/agent_step_router_v1_results.json`](../artifacts/agent_step_router_v1_results.json).
It contains task-level terminal outcomes, costs, paired statistics, mechanism
audit, protocol and grader hashes, and classified malformed-patch failures.
It excludes credentials, hidden reasoning, raw trajectories, patches, and
terminal logs.
