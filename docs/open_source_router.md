# Open-source model router guide

Budget Router can train a prompt-level model selector for any candidate pool.
It does not require Tinker, OpenAI, Anthropic, Gemini, or another specific
provider. Training consumes evaluation outcomes you already collected; routing
itself makes no model call.

It also exposes a step-level agent interface. That router is invoked before
every LLM call with the visible trajectory prefix, so a cheap model can handle
routine exploration while a stronger model handles states classified as
risky. The interface supports that behavior; the included held-out experiment
is also a warning that a trained policy can conservatively collapse to the
strong model under distribution shift.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[ml,proxy]"
```

The `ml` extra provides the shared task-model classifier and sklearn-compatible
hashing. The `proxy` extra provides the OpenAI-compatible server.

## 1. Define the model pool

Model cards are JSON. Names are opaque identifiers; providers can be hosted
APIs, local servers, fine-tunes, or complete agent workflows.

```json
{
  "models": [
    {
      "name": "local/cheap-model",
      "provider": "local",
      "expected_cost_usd": 0.002,
      "input_cost_per_million_usd": 0.05,
      "output_cost_per_million_usd": 0.15,
      "context_tokens": 32768,
      "capabilities": ["text", "json"]
    },
    {
      "name": "cloud/strong-model",
      "provider": "cloud",
      "expected_cost_usd": 0.05,
      "input_cost_per_million_usd": 2.0,
      "output_cost_per_million_usd": 8.0,
      "context_tokens": 131072,
      "capabilities": ["text", "json", "tools"]
    }
  ]
}
```

When per-token prices are present, live decisions estimate request cost from
the supplied input/output token estimates. Training-mean realized cost is used
for calibration policy selection.

## 2. Supply a complete outcome matrix

Outcomes are JSONL with one row per request-model pair:

```json
{"request_id":"train-1","text":"Summarize this paragraph.","model":"local/cheap-model","acceptable":true,"split":"train","cost_usd":0.001,"group":"summarization"}
{"request_id":"train-1","text":"Summarize this paragraph.","model":"cloud/strong-model","acceptable":true,"split":"train","cost_usd":0.030,"group":"summarization"}
```

Required fields:

- `request_id`: stable ID shared by every candidate's outcome;
- `text`: exactly what will be available when routing;
- `model`: one model-card name;
- `acceptable`: your task-specific success label;
- `split`: `train` or `calibration`.

Optional fields:

- `cost_usd`: realized request or workflow cost;
- `group`: dataset, product surface, repository, or other aggregation cohort.

Every request must have one outcome for every candidate model. A complete
matrix avoids changing the training distribution based on which model happened
to be called. If you have a continuous score, convert it to `acceptable` using
a threshold chosen before test evaluation.

Use representative application evaluations. For an agent, one row can
represent an entire officially graded trajectory, but the routing text and
cost must reflect what is available before that trajectory starts.

## 3. Train and freeze

```bash
budget-router semantic-train \
  --outcomes examples/outcomes.example.jsonl \
  --model-cards examples/model_cards.example.json \
  --output outputs/semantic_router.json
```

Training fits:

- independent prompt classifiers for each model;
- a shared prompt-model classifier with model-specific interactions;
- per-model isotonic probability calibration;
- a frozen cost weight selected on calibration data only.

The default selection rule compares paired calibration outcomes with the
calibration-best fixed model. It retains only candidates whose one-sided 95%
lower quality bound clears the configured non-inferiority margin, then chooses
the lowest-cost survivor. If no candidate passes, the artifact records
`gate_passed: false`. At runtime, only the exact classifier-method and
cost-weight candidates that passed this gate may route by default.
Unevaluated overrides and `cost` mode use the calibration-best fixed model.
If a hard constraint excludes that fallback, the router fails closed.
`allow_ungated_policy=True` is an explicit experimental override.

Artifacts include model cards, weights, calibrators, selector candidates,
training counts, and a canonical hash. Loading fails if the JSON is modified
without recomputing a valid artifact.

Keep a third, untouched test split outside this command. Evaluate the frozen
artifact exactly once against every fixed model and preserve the result,
including failures.

```bash
budget-router semantic-evaluate \
  --artifact outputs/semantic_router.json \
  --outcomes examples/heldout.example.jsonl \
  --output outputs/semantic_evaluation.json
```

The report includes every fixed model and each requested routing mode, with
success, cost, group-macro success, route counts, and paired bootstrap
intervals against the best fixed model. It also records the input file hash,
canonical outcome-matrix hash, evaluation settings, per-request selections,
policy-gate scope, and report hash. Held-out rows need `input_tokens` and
`output_tokens` when model cards use token prices; global CLI estimates are an
alternative. The default evaluation runs `quality` and `balanced`. Add
`--mode cost --allow-ungated-policy` only when an explicitly labeled ungated
cost analysis is intended.

## 4. Inspect a decision

```bash
budget-router semantic-route \
  --artifact outputs/semantic_router.json \
  --text "Debug a subtle concurrent state-machine race" \
  --mode balanced \
  --require-capability tools \
  --input-tokens 2000 \
  --output-tokens 1000 \
  --max-cost-usd 0.10
```

The result contains:

- selected model;
- calibrated success prediction for every model;
- request-level cost estimate;
- utility for every eligible model;
- rejected models and exact reason codes;
- exact policy-gate status, scope, and fallback status;
- artifact hash.

Modes:

- `quality`: highest predicted success, with lower cost as tie-breaker;
- `balanced`: predicted success minus a calibrated normalized-cost penalty;
- `cost`: cheapest model within a caller-supplied predicted-quality band;
  this mode is ungated by the default trainer and therefore falls back unless
  explicitly overridden.

Hard filters apply before a mode:

- `--allowed-model`;
- `--require-capability`;
- `--deny-provider`;
- context-window check from token estimates;
- `--max-cost-usd`.

Python usage:

```python
import json

from budget_router import SemanticRouter

with open("outputs/semantic_router.json") as stream:
    router = SemanticRouter(json.load(stream))

decision = router.route(
    "Extract invoice line items as JSON",
    mode="balanced",
    required_capabilities={"json"},
    input_tokens=2_000,
    output_tokens=800,
)
print(decision.selected_model)
```

## 5. Route individual agent calls

There are three compatible step-routing paths:

- `guarded-agent-step-router-artifact-v2` is the current frozen two-tier
  classifier. It adds initial-strong and maximum-cheap-burst guards to the
  context guard and anti-oscillation dwell rule.
- `agent-step-router-artifact-v1` is the historical unguarded two-tier
  artifact retained for reproducibility.
- `semantic-router-artifact-v1` can contain any number of model cards. At
  every step, the same `agent-route` command renders the visible trajectory,
  predicts success for every model, applies capability/provider/context/cost
  constraints, and selects according to `quality`, `balanced`, or `cost`
  mode.

An arbitrary-pool artifact must be trained on outcomes for the exact pool it
will route. The reusable adapter is provider-neutral, but the controlled
dynamic evaluation in this repository validates the frozen two-tier policy;
it does not establish that every newly supplied model pool will outperform
its fixed baselines.

The first controlled evaluation exposed a zero-activation failure: its router
selected Qwen for all 1,129 calls. The follow-up calibrated on those visible
prefixes without using their terminal outcomes, added deterministic guards,
and tested a frozen artifact on 60 new tasks. It resolved 26/60 versus fixed
Qwen's 24/60 while saving 20.79%, with GPT-OSS handling 12.70% of calls. On
the 50-task novel-repository subgroup it tied Qwen at 22/50 while saving
20.80%. The quality difference remains uncertain; see the
[guarded agent-step final report](agent_step_router_v2_final_report.md).

The portable agent artifact contains opaque cheap/strong model IDs, hashed
logistic weights, probability calibration, a threshold, context guard, and
anti-oscillation dwell rule. It contains no provider credential or provider
client.

For one decision, save the messages visible before the next LLM call:

```json
{
  "messages": [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the parser regression."},
    {"role": "assistant", "content": "I will inspect the parser."},
    {"role": "tool", "content": "<returncode>0</returncode><output>...</output>"}
  ]
}
```

Then inspect the selection without calling either model:

```bash
budget-router agent-route \
  --artifact outputs/semantic_router.json \
  --messages examples/agent_messages.example.json \
  --step-index 0 \
  --mode balanced
```

For a real agent loop, keep one arbitrary-pool router object for the whole
trajectory:

```python
import json

from budget_router import SemanticAgentRouter

with open("outputs/semantic_router.json") as stream:
    router = SemanticAgentRouter(json.load(stream))

while not agent.finished:
    decision = router.select(
        agent.visible_messages,
        mode="balanced",
        required_capabilities=agent.required_capabilities,
        output_tokens=agent.expected_output_tokens,
        max_cost_usd=agent.remaining_call_budget,
    )
    response = providers[decision.selected_model].complete(agent.visible_messages)
    agent.observe(response)
```

This returns predictions and estimated costs for the whole pool, exact
rejection reasons, the artifact hash, a bounded prefix summary, and whether
the route switched models. It does not call a provider itself.

The guarded two-tier artifact used in the published experiment is local
research output and is not included in source distributions. Its schema and
runtime classes remain supported for reproducibility; the generic semantic
artifact above is the public quickstart.

The provider map can contain hosted APIs, local models, fine-tunes, or
different agent endpoints. The included Tinker mini-swe adapter is one
integration, not a runtime requirement.

Training labels must be conditional step labels: for each visible prefix,
identify the cheapest tier that still preserves the task-level pass predicate.
The included TwinRouterBench training script demonstrates execution-verified
tier supervision and instance-grouped train/calibration/test splits. When
building a new pool, regenerate labels for that pool; tier labels are not
timeless facts about prompts.

Use the same public renderer when constructing a prefix-by-model outcome
matrix:

```python
from budget_router import render_agent_prefix

training_text, summary = render_agent_prefix(
    agent.visible_messages,
    step_index=agent.model_calls,
)
```

Evaluate every candidate model or agent endpoint from that frozen prefix,
store its terminal success and cost under the same `training_text`, then train
the arbitrary-pool semantic artifact from the complete matrix. Group splits by
task or trajectory so prefixes from one run cannot leak across training and
evaluation.

## 6. Serve an OpenAI-compatible endpoint

Map every model name to an OpenAI-compatible upstream:

```json
{
  "artifact": "../outputs/semantic_router.json",
  "mode": "balanced",
  "router_model_name": "auto",
  "default_output_tokens": 1024,
  "allow_ungated_policy": false,
  "upstreams": {
    "local/cheap-model": {
      "base_url": "http://127.0.0.1:8001/v1",
      "api_key_env": "LOCAL_MODEL_API_KEY",
      "upstream_model": "local-model"
    },
    "cloud/strong-model": {
      "base_url": "https://provider.example/v1",
      "api_key_env": "CLOUD_MODEL_API_KEY",
      "upstream_model": "strong-model"
    }
  }
}
```

Credentials are never stored in the config. `api_key_env` names an environment
variable. Static authorization or API-key headers are rejected.

```bash
export LOCAL_MODEL_API_KEY="..."
export CLOUD_MODEL_API_KEY="..."

budget-router serve-proxy \
  --config examples/proxy.example.json \
  --host 127.0.0.1 \
  --port 8000
```

Use any OpenAI-compatible client, or call the endpoint directly:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain this transaction anomaly."}]
  }'
```

The proxy supports normal and streaming chat completions. It adds:

- `x-budget-router-model`;
- `x-budget-router-predicted-success`;
- `x-budget-router-artifact`.

`POST /v1/router/decision` returns the decision without calling an upstream.
`GET /health` returns the artifact hash and model pool.

## Production checklist

Before sending live traffic:

1. Make train, calibration, and test groups independent at the unit that can
   leak—customer, repository, conversation, or time period.
2. Compare the frozen policy against every fixed candidate.
3. Add an explicit frontier fallback for unsupported or novel workloads.
4. Shadow route first; log decisions, selected-model outcomes, latency, cost,
   and artifact hash.
5. Monitor calibration, route share, model errors, provider errors, input
   drift, and realized budgets separately.
6. Keep provider health/failover below semantic model selection.
7. Revalidate after model, prompt, harness, evaluator, price, or traffic
   changes.
8. Preserve session consistency for multi-turn workloads unless step-level
   switching is separately evaluated.
9. Before a paid step-routing test, require a dynamic shadow-activation gate;
   a static held-out cheap share does not guarantee any cheap calls in the
   live agent.

The sanitized prompt-study record demonstrates why this matters: the balanced
classifier matched GPT-5 on its held-out ID workload while saving 32.22%, but
failed severely on an unseen ARC-AGI workload. The first agent-step artifact
failed in the opposite direction by routing every live state to the strongest
model. Representative live-prefix calibration repaired activation in the
guarded follow-up, but neither result makes calibration timeless: revalidate
whenever the model pool, harness, prompts, tools, evaluator, prices, or
workload changes.
