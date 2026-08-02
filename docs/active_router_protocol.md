# Active protocol: classifier-centered open model router

Status: **normative; completed through Amendment 013; further Tinker
strong-candidate search stopped**

Study ID: `open-router-study-2026-07-29-v1`

Machine-readable lock: `artifacts/active_router_protocol.json`

Active amendments:

- `docs/active_router_protocol_amendment_001_frontier_references.md`
- `docs/active_router_protocol_amendment_002_nemotron_tool_recovery.md`
- `docs/active_router_protocol_amendment_003_context_budgeting.md`
- `docs/active_router_protocol_amendment_004_docker_image_preflight.md`
- `docs/active_router_protocol_amendment_005_public_matrix.md`
- `docs/active_router_protocol_amendment_006_arc_agi_replication.md`
- `docs/active_router_protocol_amendment_007_agent_step_routing.md`
- `docs/active_router_protocol_amendment_008_guarded_agent_step_followup.md`
- `docs/active_router_protocol_amendment_009_three_tier_quality_cost.md`
- `docs/active_router_protocol_amendment_010_task_block_image_lifecycle.md`
- `docs/active_router_protocol_amendment_011_kimi128_strong_qualification.md`
- `docs/active_router_protocol_amendment_012_terminal_workspace_capture.md`
- `docs/active_router_protocol_amendment_013_mit_license.md`

This document supersedes earlier proposals as the active plan. Earlier
experiments and protocols remain evidence and design history; they do not
authorize new paid calls.

## 1. Deliverables

The project has two inseparable deliverables:

1. A technical blog post explaining what public production routers reveal,
   how this router was developed, which alternatives were tested, and how it
   performed against complete fixed-model baselines.
2. An MIT-licensed, provider-neutral router that accepts arbitrary model cards
   and logged outcomes, trains a calibrated classifier, exposes cost/quality
   controls, and can be used as a Python library or OpenAI-compatible proxy.

The core router is general-purpose. SWE-bench is the controlled agentic case
study, not a restriction baked into the library.

## 2. Scientific claim and routing paradigm

The primary hypothesis is:

> A calibrated task-model classifier can match or improve the best fixed model
> at lower observed cost by assigning cheap models only where they are
> predicted to preserve quality.

For each eligible model \(m\), the semantic router estimates

\[
\hat p_m = P(\text{acceptable outcome} \mid
  \text{request}, \text{context}, \text{model}, \text{visible state}).
\]

Hard capability, privacy, context, tool, and price constraints are applied
before scoring. The decision layer supports:

- `quality`: maximize calibrated success probability;
- `balanced`: maximize success utility after cost, latency, and switching
  penalties;
- `cost`: choose the cheapest model whose calibrated probability clears a
  quality threshold.

This semantic layer is separate from provider routing. Provider health,
latency, service tier, fallback, and cache/session stickiness are separate
modules and separate experiments.

The primary learned treatment is classifier-based. Heuristics, retrieval,
gradient boosting, and temporal classification are comparators or ablations;
they do not replace the public classifier interface.

## 3. Evidence informing the design

The blog and implementation will distinguish public facts from our
interpretation.

- Cursor publicly describes a pre-inference classifier using query, context,
  complexity, domain, and model behavior, trained on more than 600,000 live
  requests and evaluated online with cache effects included:
  <https://cursor.com/blog/router>.
- OpenRouter Auto Beta classifies prompts into roughly 30 task types, combines
  that classification with trailing community model usage, applies a
  cost-quality control, and uses session stickiness:
  <https://openrouter.ai/docs/guides/routing/routers/auto-router>.
- OpenRouter explicitly separates model selection from provider selection:
  <https://openrouter.ai/blog/insights/model-routing/>.
- Ramp describes a lowest-cost-model-above-quality-bar product and, in a
  separate infrastructure write-up, online Thompson sampling over failures
  and latency:
  <https://ramp.com/router/> and
  <https://builders.ramp.com/post/thompson-sampling-model-routing>.
- Not Diamond documents pretrained and application-specific custom routers,
  including arbitrary custom inference endpoints with evaluation data:
  <https://docs.notdiamond.ai/docs/key-concepts> and
  <https://docs.notdiamond.ai/docs/routing-between-custom-models>.
- vLLM Semantic Router places a BERT-style semantic layer above a separate
  production-routing stack:
  <https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md>.
- RouteLLM supplies the learned weak/strong classifier precedent:
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html>.
- SWE-Router motivates the temporal ablation using partial agent
  trajectories: <https://arxiv.org/abs/2607.00053>.
- LLMRouterBench supplies a public, broad, multi-model evaluation corpus:
  <https://github.com/ynulihao/LLMRouterBench>.

## 4. What prior local work contributes

Prior paid outcomes are reused where compatible; they are never relabeled as
new evidence.

- Qwen 3.6 35B-A3B is the proven control.
- Qwen 3 8B resolved only 1 of 32 screen tasks and is not rescreened.
- Inkling and Kimi K2.6 were strictly below Qwen on the completed held-out
  matrix, with every success a subset of Qwen's successes.
- The separately qualified 128K Kimi K2.6 treatment was structurally valid on
  12/12 frozen tasks but submitted only 6 patches and formally resolved 4/12.
  A post-hoc infrastructure audit recovered two complete tracked diffs from
  visible terminal outputs; both passed the official grader. The separately
  labeled terminal-workspace diagnostic is therefore Kimi 6/12 versus fixed
  Qwen's 6/12, with exactly the same solve set. Kimi still missed the frozen
  8/12 threshold, added no unique solve, and cost 8.26 times as much.
  Amendments 011 and 012 therefore retain the decision to stop further Tinker
  strong-candidate search without claiming that Kimi was less capable on this
  cohort.
- Nemotron 3 Super was structurally valid but slow and weak in the completed
  pilot.
- The old GPT-OSS 120B treatment was the 128K `:peft:` configuration and
  submitted no patches. The current 32K non-PEFT treatment is a different,
  cheaper treatment and may be screened once.
- Existing same-workspace and isolated-scout studies are temporal pilot
  evidence. They are not the final temporal evaluation.

Reusable infrastructure includes the official SWE-bench grader integration,
Tinker runner, exact cost ledger, trace schema, task-aware hashed logistic
baseline, repository-aware split tooling, and frozen-artifact checks.

## 5. Tinker candidate screen

Catalog and list prices are frozen from Tinker's public catalog as viewed on
2026-07-29: <https://tinker-docs.thinkingmachines.ai/tinker/models/>.
Prices are per million tokens.

### Proven control

| Role | Model | Family | Context | Prefill / sample | Episode cap |
|---|---|---|---:|---:|---:|
| medium control | `Qwen/Qwen3.6-35B-A3B` | Qwen hybrid MoE | 64K | $0.54 / $1.335 | $0.90 |

Existing official outcomes on development task IDs are reused. It is run
afresh on every held-out task.

### Cheap and medium candidates

| Model | Why it is screened | Context | Prefill / sample | Screen cap |
|---|---|---:|---:|---:|
| `openai/gpt-oss-20b` | cheap OpenAI reasoning family; useful potential scout or whole-task tier | 32K | $0.18 / $0.45 | $0.35 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | cheap NVIDIA hybrid family optimized for agentic reasoning | 64K | $0.195 / $0.495 | $0.35 |
| `openai/gpt-oss-120b` | current non-PEFT medium reasoning treatment; tests within-family scale | 32K | $0.33 / $0.84 | $0.75 |

These run on 12 development tasks, stratified across development repositories
and enriched for both Qwen successes and failures. The screen is for treatment
selection, not an unbiased performance estimate.

### Higher-cost challengers

| Model | Why it is screened | Context | Prefill / sample | Screen cap |
|---|---|---:|---:|---:|
| `deepseek-ai/DeepSeek-V3.1` | distinct hybrid family with documented multi-turn tool training | 32K | $1.695 / $4.215 | $1.50 |
| `Qwen/Qwen3.6-27B` | dense coding-oriented challenger; tests whether the best fixed model is above the current control | 64K on Tinker | $1.86 / $5.595 | $1.50 |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | distinct frontier-scale NVIDIA agentic reasoning family | 64K | $2.49 / $6.225 | $1.50 |

These run on eight development tasks enriched with Qwen failures, plus enough
Qwen successes to detect destructive behavior. This is deliberately a
complementarity screen, not an accuracy estimate.

### Excluded candidates

The following are not rescreened without a versioned amendment:

- Qwen 3 8B and Qwen 3.5 4B: prior weak or structurally invalid treatments;
- Inkling and Kimi K2.6: completed held-out successes were subsets of Qwen;
- Nemotron 3 Super: prior weak, very slow treatment;
- Qwen 3.5 397B: same-family, more expensive redundancy while Qwen 27B is the
  coding-oriented dense challenger;
- base models: not instruction-tuned treatments for this agent harness;
- extended-context `:peft:` variants: unnecessary cost for the frozen task
  context and not interchangeable with their standard treatments.

## 6. Candidate gates and final pool

All screen submissions receive official terminal grades before selection.
Provider or harness failures are not model-quality failures; they trigger the
predeclared infrastructure repair rule.

Minimum eligibility:

- cheap/medium candidates: structurally valid on at least 10 of 12 tasks;
- higher-cost candidates: structurally valid on at least 7 of 8 tasks;
- no unresolved systematic renderer or tool-protocol defect;
- at least one official resolution on the screen.

The final Tinker pool contains at most four models:

1. the Qwen 3.6 35B control;
2. one cheap winner;
3. one challenger from a family distinct from the control and cheap winner;
4. optionally, one additional non-dominated or uniquely complementary model.

The cheap winner is chosen by, in order:

1. more official resolutions;
2. more tasks resolved at lower cost than Qwen when both resolve;
3. lower total observed cost;
4. lower latency.

A challenger is retained only if it does at least one of:

- resolves a task missed by every cheaper eligible candidate;
- has the highest screen resolution count;
- lies on the observed resolution/cost Pareto frontier.

The optional fourth model requires at least two unique screen resolutions or a
non-dominated cost/quality point that changes the observed frontier. The final
pool must contain at least three model families. If no diverse routing
opportunity survives, paid matrix expansion stops; the project continues with
the public benchmark and an honest negative Tinker case study.

After the complete 24-task development matrix, the same dominance test is
applied once more. Models may be pruned before the held-out freeze, but no new
model may be added.

## 7. Development and held-out design

### Public broad study

LLMRouterBench is used for the high-sample general routing study. We evaluate
fixed models, cheapest, best-single, random, task-family lookup, prompt-only
logistic, embedding k-nearest-neighbor, gradient boosting, shared task-model
classification, and the hindsight oracle. We report ID and held-out-domain
results, calibration, cost at matched quality, quality at matched cost,
Pareto distance, route entropy, and bootstrap intervals.

No Tinker spend is required for this stage.

### Tinker development matrix

- 24 development tasks.
- Development repositories are disjoint from prospective held-out
  repositories.
- Tasks are chosen only from public inputs with existing reusable Qwen control
  outcomes where possible.
- All surviving candidate rows are completed so every development task has an
  official outcome for every final model.
- Model selection, classifier choice, calibration, thresholds, feature
  ablations, and temporal-gate fitting use development data only.

### Tinker held-out matrix

- 20 previously unexecuted exact SWE-bench Verified task IDs.
- Repository-disjoint from this study's development cohort.
- Every final model is run on every task with the frozen harness, renderer,
  seed policy, and per-treatment cap.
- The task list, classifier artifacts, thresholds, and router actions are
  frozen before any held-out inference.
- All fixed-model and temporal episodes finish before any official held-out
  grade is opened.

This is a prospective exact-task holdout, but not a repository-naive study:
earlier project experiments used the finite SWE-bench Verified repository
pool. The blog must state that limitation.

## 8. Classifiers and controls

Required controls:

- each fixed model;
- cheapest eligible fixed model;
- development-best fixed model;
- frequency-matched random router;
- task-family/rule lookup;
- hindsight task oracle, clearly labeled non-deployable.

Required learned methods:

- per-model hashed logistic heads;
- shared task-model classifier using task and model-card features;
- embedding k-nearest-neighbor;
- calibrated gradient-boosted classifier.

All hyperparameters and cost/quality thresholds are selected using
leave-one-development-repository-out predictions. The primary frozen router is
the simplest learned method on the development Pareto frontier, breaking ties
by:

1. higher cross-validated resolution;
2. lower cross-validated cost;
3. better calibration;
4. lower implementation complexity.

Test labels may not select the winning method.

Primary comparison:

- cost savings at the development-best fixed model's held-out resolution
  count.

Secondary comparisons:

- held-out resolution at matched observed cost;
- total and mean cost;
- cost per resolution;
- Brier score and reliability;
- regret to the oracle;
- route distribution and uncertainty;
- paired bootstrap intervals and exact paired tests.

No positive headline is permitted unless the router matches or exceeds the
best fixed model's held-out resolution count while reducing observed total
cost. Otherwise the result is reported as mixed or negative.

## 9. Temporal arm

The temporal hypothesis is tested separately from static selection:

> Visible evidence from a cheap model's first few agent turns improves the
> decision to continue cheaply or escalate, compared with issue text alone.

Training uses the final cheap model's truncated development trajectories,
corresponding fixed-Qwen development outcomes, existing local cascade traces,
and a separate reproduction on the public SWE-Router data.

Before held-out inference, freeze:

- checkpoint at no more than six cheap-model calls;
- cheap scout sub-cap of at most $0.08;
- visible features and sanitizer;
- classifier and threshold;
- same-workspace handoff;
- total per-task cap of $0.90;
- action set: continue cheap or hand off to Qwen 3.6 35B.

The on-policy temporal arm runs on the same 20 held-out task IDs before grades
are opened. It is compared against fixed cheap, fixed Qwen, the static router,
and a non-deployable temporal oracle. Switching and cache-miss costs are
included.

## 10. Incremental USD budget

The new authorization is a hard incremental ceiling of **$220.00** beginning
after this protocol is frozen. It does not retroactively include prior
experiments.

| Ordered stage | Maximum incremental exposure |
|---|---:|
| synthetic protocol smoke for all candidates | $1.00 |
| 12-task cheap/medium screen | $17.40 |
| 8-task higher-cost challenger screen | $36.00 |
| complete the 24-task final-pool development matrix | $52.20 |
| 20-task held-out fixed-model matrix, at most four models | $85.00 |
| 20-task frozen temporal arm | $18.00 |
| unallocated contingency | $10.40 |
| **absolute total** | **$220.00** |

Each stage starts only if its complete maximum reservation plus all protected
future-stage reserves fits the remaining authorization. Actual spend below a
cap remains unspent. It may not silently expand sample size, add models, add
seeds, or fund post-hoc variants.

If candidate pruning reduces later maximum exposure, the freed amount remains
reserved for grader recovery, infrastructure failures, or an explicit
versioned amendment. It is not a target to consume.

## 11. Provider-neutral open-source implementation

The public interface will be independent of Tinker:

```python
router = ModelRouter.fit(
    outcomes="evaluations.jsonl",
    models="models.yaml",
)

decision = router.route(
    messages=[{"role": "user", "content": "Fix this parser regression"}],
    allowed_models=["small", "medium", "strong"],
    objective="balanced",
)

router.observe(
    decision.id,
    quality=1.0,
    cost_usd=0.03,
    latency_ms=840,
)
```

Required release features:

- versioned model-card and outcome schemas;
- arbitrary model IDs and executor callbacks;
- a generic OpenAI-compatible provider adapter;
- local classifier training, calibration, evaluation, and artifact export;
- quality, balanced, and cost modes;
- hard eligibility constraints;
- session stickiness and cache/switch accounting;
- separate provider health and fallback policy;
- temporal checkpoint API;
- explanation, uncertainty, and observability;
- an OpenAI-compatible proxy command;
- examples using local models, hosted providers, and precomputed outcomes.

Cold-start behavior must be explicit: a model without outcomes is either
excluded, routed through a configured prior, or explored under a user-selected
budget. The project will not claim zero-data optimal routing.

## 12. Blog structure

Working title:

> The Classifier Is Only the Beginning: Building an Open Model Router from
> Production Lessons

Required sections:

1. What Cursor, Ramp, OpenRouter, Not Diamond, and vLLM publicly reveal.
2. Semantic selection versus provider/runtime routing.
3. What our earlier experiments got wrong and what they still taught us.
4. Completing diverse fixed-model baselines.
5. Comparing classifier paradigms and simple controls.
6. Prompt-only versus partial-trajectory routing.
7. Cost-quality Pareto results with uncertainty.
8. The provider-neutral open-source architecture and quickstart.
9. Limitations, distribution shift, model churn, cache effects, and cold
   starts.

Company performance numbers must be labeled self-reported. Public benchmark
reproductions and our Tinker experiment must be presented separately.

## 13. Change control and stop rules

This protocol is deliberately adaptive only at its predeclared candidate
gates. Those gates are not protocol drift.

Any other change requires:

1. a documented reason;
2. a versioned amendment made before collecting affected outcomes;
3. an updated machine-readable artifact and SHA-256;
4. explicit user approval if the change alters the research question, held-out
   design, model-selection rule, or USD ceiling.

Allowed infrastructure repairs do not include prompt or cap tuning based on
model quality. After held-out grades are opened, no new method may be called
confirmatory on those tasks.

Stop paid work immediately if:

- the cumulative incremental reservation can exceed $220;
- a credential or secret would enter an artifact or task container;
- an unresolved infrastructure defect invalidates treatment comparability;
- fewer than three model families pass the screen;
- the final development matrix has no observed routing opportunity beyond the
  best fixed model.

Stopping a paid stage does not stop the public-data study or open-source
implementation.
