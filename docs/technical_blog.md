# Routing models inside coding agents

*How a calibrated per-call policy cut conservative SWE-bench cost by 20.8%
at the same observed pass rate.*

*Pranav Dulepet*

On 50 held-out SWE-bench tasks from repository families excluded from fitting,
our router and fixed Qwen3.6 35B each resolved 22. Conservative cost was
$20.61 for the router and $26.02 for Qwen. This accounting prices every input
token at the uncached rate for both policies.

Across all 60 tasks, the router resolved 26 and fixed Qwen resolved 24.
Conservative cost was $24.54 and $30.98. The quality interval remained wide:
−6.67 to +13.33 percentage points.

The project moved through three stages:

| Stage | System | Result |
|---|---|---|
| Prompt routing | Shared request-model classifier over 13 models | 73.64% vs GPT-5’s 73.52% at 32.2% lower benchmark cost; secondary evidence |
| Agent adaptation | Classifier over visible prefixes with explicit guards | GPT-OSS handled 12.70% of held-out agent calls |
| Frozen agent evaluation | Router, fixed Qwen, and fixed GPT-OSS on 60 tasks | 26/60 vs Qwen’s 24/60 at 20.8% lower conservative cost |

The result supports lower cost at an observed same-or-better pass rate. It
does not establish that the router is more accurate.

## Two routing problems

A prompt router makes one choice before generation. An agent router repeats
that choice throughout a trajectory. The second problem is harder because
each answer changes the next state: commands, tool output, context length,
working tree, and remaining budget.

I separate both from provider routing.

**Model routing** asks which model is most likely to solve the visible task at
an acceptable cost. **Provider routing** asks where to execute that model:
which endpoint is healthy, fast, available in-region, and likely to preserve
session caches.

The routing core, CLI, and decision API return a model without calling a
provider. The optional proxy executes that decision. A runtime layer can add
health checks, retries, and failover.

![Two-stage architecture: offline training produces a calibrated artifact; online routing filters models, predicts success, applies a cost-quality policy, and passes the selected model to a separate provider layer.](assets/router-architecture.svg)

*The semantic router selects a model. Provider health and execution remain a
separate policy.*

## What production routers reveal

Their implementations differ. Their public descriptions share one boundary:
model selection is separate from execution.

| System | Semantic decision | Runtime decision | Public lesson |
|---|---|---|---|
| [Cursor Router](https://cursor.com/blog/router) | Classifies task, domain, and complexity using live request data | Accounts for cache state | A classifier needs product modes, controls, online evaluation, and cache awareness |
| [OpenRouter Auto](https://openrouter.ai/docs/guides/routing/routers/auto-router) | Classifies requests into task categories, then applies a cost-quality preference | Keeps [model and provider routing](https://openrouter.ai/blog/insights/model-routing/) distinct | Semantic selection and infrastructure selection should not be conflated |
| [Not Diamond](https://docs.notdiamond.ai/docs/key-concepts) | Trains custom routers on application evaluations | Connects the selected model or agent endpoint | Representative outcome data matters more than a generic “difficulty” label |
| [Ramp](https://builders.ramp.com/post/thompson-sampling-model-routing) | Starts from a caller’s model preference order | Learns provider failure and latency online | Availability, deadlines, and latency require a separate adaptive layer |
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | Combines domain, complexity, tool, privacy, and safety signals | Integrates with a production stack for serving | Signals, policy, and deployment bindings should be independently inspectable |

Cursor reports training from more than 600,000 live requests and testing over
millions. Its reported savings are company measurements, not our results.
Ramp similarly reports a 26.3% cost reduction with a −0.09 percentage-point
error-rate change in its production experiment.

OpenRouter’s current Auto Beta is also distinct from its deprecated
Not Diamond-powered auto router. The current system classifies requests into
roughly 30 task types, considers recent model usage, and exposes a
cost-quality control.

Our implementation follows that separation. It provides calibrated semantic
selection, hard constraints, inspectable artifacts, agent-prefix rendering,
and an OpenAI-compatible proxy. It does not replace provider-health routing
or online drift detection.

The classifier paradigm is not new. This project contributes a
provider-neutral implementation and traces the idea from one-shot prompts to
individual agent calls. It also audits whether the intended mechanism
actually activated.

[Shnitzer et al.](https://arxiv.org/abs/2309.15789) formulate multi-model
routing as binary prediction, while
[RouteLLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html)
learns strong-versus-weak routing and evaluates quality-cost tradeoffs.

## The classifier

For request or agent state \(x\) and model \(m\), the router estimates

\[
\hat p_m(x)=P(\text{acceptable outcome}\mid x,m).
\]

The generic open-source trainer expects a complete outcome matrix: the same
requests evaluated by every candidate model. It learns shared text features,
a bias for each model, and prompt-feature × model interactions. Shared terms
represent patterns that are generally easy or hard. Interactions represent
model-specific strengths.

Calibration-only outcomes convert raw scores into per-model probabilities.
The balanced policy then selects

\[
m^*(x)=\arg\max_m\left[
\hat p_m(x)-\lambda
\frac{\mathbb E[C_m]}{\max_j \mathbb E[C_j]}
\right].
\]

The cost penalty \(\lambda\) moves the decision along a quality-cost curve.
Quality mode ignores the penalty. Cost mode finds the cheapest model within a
configured predicted-quality distance of the best candidate.

Hard filters run first. They can exclude a model by capability, provider,
context window, allowed list, or request-cost ceiling. If calibration fails
the non-inferiority gate, the exact failed or unevaluated policy falls back to
the best fixed calibration model. Cost mode is ungated by the default trainer
unless a caller explicitly overrides that safeguard.

A single “easy, medium, hard” label would lose model-specific strengths. The
router instead estimates a success probability for every request-model pair.
A request can favor one model for code and another for factual retrieval.

## Prompt routing

We first used the pinned public LLMRouterBench release. It supplied 151,554
outcomes: 11,658 prompts evaluated by 13 models. The pool covered GPT,
Claude, Gemini, DeepSeek, Qwen, Kimi, GLM, and Intern families at different
prices. Seven datasets formed deterministic train, calibration, and
in-distribution test splits. Three ArenaHard partitions supplied an
out-of-distribution test.

The study froze the model pool, complete-matrix rule, splits, outcome
threshold, cost field, fixed baselines, learned methods, cost weights, and
paired bootstrap before test scoring.

The published balanced artifact is the shared classifier at
\(\lambda=0.10\). It scored 73.64% and cost $38.31, compared with GPT-5 at
73.52% and $56.52. Its paired quality interval was −1.54 to +2.02 points;
its cost-saving interval was 28.64% to 35.92%.

That point is secondary evidence because it was selected from the frozen
curve after test scoring. The prespecified text k-nearest-neighbor primary
scored 65.93%, below fixed GPT-5 at 73.52%. The study therefore does not make
a confirmatory prompt-routing claim.

![Quality-cost curves for four prompt-routing classifiers on 2,184 held-out prompts.](assets/public-quality-cost.svg)

*Each line varies only the frozen cost penalty. The hindsight oracle uses
terminal outcomes and is not deployable.*

The oracle reached 87.70%, 14.18 points above fixed GPT-5, at $8.96 rather
than $56.52. The learned routers recovered only part of that gap.
[LLMRouterBench](https://aclanthology.org/2026.findings-acl.1881/) reports the
same broad problem: simple baselines remain strong, model recall is difficult,
and larger pools do not automatically produce better routing.

The same artifact defined the deployment boundary.

| Evaluation | Balanced router | Fixed comparator | Cost change | Interpretation |
|---|---:|---:|---:|---|
| In-distribution, 2,184 prompts | 73.64% | GPT-5: 73.52% | −32.22% | Secondary, quality compatible |
| ArenaHard, 750 prompts | 69.33% | GPT-5: 69.66% | −40.0% | GPT-5 was not the post-hoc best fixed model |
| ARC-AGI, 400 prompts | 25.25% | GPT-5: 52.00% | −98.07% | Unsupported distribution; strong fallback required |

On ARC-AGI, the frozen router selected Qwen3 235B regular for 399 of 400
prompts. Grid reasoning was absent from its training distribution. The
result marks a support boundary: without evidence for grid reasoning, the
router needed to abstain and select its strong fallback.

Production use therefore needs selective abstention: measure support for the
current input and use a configured strong fallback when support is low.
Calibration on one distribution does not cover another.

## Adapting the router to agents

Prompt outcomes do not show whether a model can operate a coding harness,
follow a tool protocol, preserve a working tree, and submit a patch. We
therefore evaluated complete treatments: model, renderer, context policy,
turn limit, cost cap, tools, and submission path.

The qualification gate retained only treatments that were structurally
valid, useful, and cost-competitive. It evaluated the model, renderer,
context policy, tools, and submission path as one arm.

The final design combined four requirements:

| Requirement | Implementation |
|---|---|
| Make decisions at agent-call granularity | Route from the visible prefix before each model call |
| Protect quality | Fall back to the calibrated strong model |
| Represent the current trajectory | Include messages, tool results, context length, and step position |
| Verify economic activation | Measure cheap-call share and trajectory coverage before launch |

## The guarded agent router

For the final policy, we reused only visible intermediate prefixes—not
terminal task outcomes. We excluded the new test repository families, refit
the classifier, calibrated a live activation threshold, and added four
guards:

1. Use Qwen for the first two calls.
2. Use Qwen after the visible context exceeds 24,000 tokens.
3. Keep Qwen for at least two calls after an escalation.
4. Allow at most two consecutive GPT-OSS calls.

The classifier was a binary logistic model over hashed text features. It
learned to choose between tiers from a public question bank with
execution-verified labels. It was not trained on the 60 held-out SWE-bench
outcomes. The guards encoded risks that the training data did not represent.

A 20-task development gate required official resolution, nonzero cheap use,
sufficient trajectory coverage, and no infrastructure failures. The unchanged
artifact passed. We then froze 60 unexecuted tasks and completed all three
treatments before opening official grades. Fifty tasks came from Astropy,
Django, and Matplotlib, repository families excluded from fitting and
threshold selection.

![Official resolutions versus conservative cost for fixed GPT-OSS, fixed Qwen, and the guarded agent router.](assets/agent-quality-cost.svg)

*The router’s cost interval excluded zero saving. Its quality interval
included losses and gains.*

| Policy | Submitted | Resolved | Conservative cost |
|---|---:|---:|---:|
| Fixed GPT-OSS 20B | 40/60 | 13/60 | $9.5290 |
| Fixed Qwen3.6 35B | 42/60 | 24/60 | $30.9780 |
| Guarded step router | 44/60 | 26/60 | $24.5376 |

Five tasks were router-only successes and three were Qwen-only. That
difference was compatible with chance (exact McNemar \(p=0.7266\)). The
paired cost-saving interval was 9.91% to 30.57%.

On the predeclared novel-repository subgroup, both policies resolved 22/50.
The router saved 20.80%, with a paired cost interval of 11.34% to 30.13%.
Its quality interval was −10 to +10 points. The exact tie covers three
repository families, not coding work in general.

The router actually used both models:

![The 2,866 agent calls partitioned by classifier and guard decisions.](assets/agent-decision-audit.svg)

*GPT-OSS handled 12.70% of calls across 56 of 60 trajectories.*

Repricing the 364 GPT-OSS calls at Qwen’s rates isolates $1.95 in direct
model-substitution savings. The total policy difference was $6.44; the rest
came from changed commands, token counts, call counts, and stopping behavior.
The paired policy comparison measures the entire trajectory. Same-token
repricing measures only substitution.

## What the evidence supports

The prompt and agent studies support four conclusions:

1. A calibrated classifier can lower cost on a held-out workload without an
   observed quality decline. The prompt result is secondary evidence; the
   agent result is the stronger application study.
2. Agent routing needs visible agent-state evidence. Activation rate belongs
   beside cost and quality in the evaluation.
3. The agent cost evidence is stronger than its quality evidence. The cost
   interval excluded zero; the quality interval did not.
4. A deployable router needs support detection, abstention, and a fixed strong
   fallback for inputs outside its calibration distribution.

SWE-bench Verified also has a limited role here. OpenAI has documented
[contamination and test-quality problems](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
and recommends newer evaluations for frontier capability. We use it as a
fixed paired workload under one scaffold, not as a current model leaderboard.
Every agent treatment has one stochastic trajectory per task. More repository
clusters and repeated seeds would narrow the inference.

## Using the open-source router

The package accepts arbitrary model IDs and providers. A user supplies model
cards, a complete train/calibration outcome matrix, and an untouched test
matrix. Training exports a content-hashed JSON artifact. Routing is local and
does not call a provider.

```bash
pip install -e ".[ml,proxy]"

budget-router semantic-train \
  --outcomes examples/outcomes.example.jsonl \
  --model-cards examples/model_cards.example.json \
  --output outputs/semantic_router.json

budget-router semantic-route \
  --artifact outputs/semantic_router.json \
  --text "Debug a concurrent state-machine race" \
  --mode balanced

budget-router semantic-evaluate \
  --artifact outputs/semantic_router.json \
  --outcomes examples/heldout.example.jsonl \
  --output outputs/semantic_evaluation.json
```

The same artifact can route a visible agent trajectory with `agent-route` or
`SemanticAgentRouter`. The OpenAI-compatible proxy maps the returned model ID
to arbitrary compatible upstreams. The proxy config stores credential
environment-variable names; secret values remain in the environment. The
artifact contains neither.

The [README](../README.md) contains the self-contained quickstart, schemas,
Python APIs, constraints, proxy configuration, gate behavior, and release
checks. The [production guide](open_source_router.md) covers larger model
pools and agent integration.

The code is provider-neutral. The evidence is not universal. Users still need
representative outcomes for their own models, prompts, harness, prices, and
success definition. A model without outcomes is excluded; the package does
not claim zero-data optimal routing.

## The next experiment

I would keep the completed cohorts sealed. Retuning on their official labels
would convert the holdout into training data.

The next study should combine task-level and step-level decisions:

1. Freeze a newer benchmark or time-disjoint repository cohort.
2. Collect a complete task × model baseline matrix with multiple trajectory
   seeds.
3. Train a task router, a per-step router, and a hierarchical router under the
   same cost accounting.
4. Add support detection and a fixed strong fallback before evaluation.
5. Open official grades only after every treatment is terminal.

This would test the original agent-routing hypothesis directly: choose the
right model for the task, then revise that choice as the trajectory changes.
On one held-out cohort, per-call substitution produced lower cost and no
observed pass-rate decline. The wide quality interval leaves losses and gains
plausible. The evaluated design paired the classifier with trajectory guards
and a fixed strong fallback. Deployment beyond its training distribution also
requires support detection.

## Reproducibility

This article is the narrative synthesis. The linked reports preserve each
experiment’s frozen design and result.

The [active protocol](active_router_protocol.md) records the ordered study,
gates, and amendments. Sanitized aggregate results are in
[`public_router_v1_results.json`](../artifacts/public_router_v1_results.json),
[`agent_step_router_v1_results.json`](../artifacts/agent_step_router_v1_results.json),
and
[`agent_step_router_v2_results.json`](../artifacts/agent_step_router_v2_results.json).
The complete agent reports are
[v1](agent_step_router_v1_final_report.md) and
[v2](agent_step_router_v2_final_report.md). Figure data and generation are
published in
[`public_router_v1_quality_cost_curves.json`](../artifacts/public_router_v1_quality_cost_curves.json)
and
[`generate_blog_figures.py`](../scripts/generate_blog_figures.py).
