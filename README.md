# Budget Router

Budget Router learns which model to use from your own evaluation results. It
returns a model ID before inference; your application remains responsible for
calling the provider.

It works with any model pool when you provide:

- a model card for each candidate;
- a complete request × model outcome matrix;
- separate train, calibration, and test requests.

It does not guess how an unseen model performs. Add new outcomes and retrain
when the model pool changes.

## Install

Python 3.12 is the reference runtime.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[ml,proxy]"
```

The core package has one runtime dependency. The `ml` extra adds
scikit-learn training. The `proxy` extra adds the OpenAI-compatible gateway.

## Quickstart

The included example trains locally and makes no provider calls.

```bash
budget-router semantic-train \
  --outcomes examples/outcomes.example.jsonl \
  --model-cards examples/model_cards.example.json \
  --output outputs/router.json

budget-router semantic-route \
  --artifact outputs/router.json \
  --text "Debug a concurrent state-machine race" \
  --mode balanced \
  --require-capability tools \
  --input-tokens 2000 \
  --output-tokens 1000

budget-router semantic-evaluate \
  --artifact outputs/router.json \
  --outcomes examples/heldout.example.jsonl \
  --output outputs/evaluation.json

budget-router agent-route \
  --artifact outputs/router.json \
  --messages examples/agent_messages.example.json \
  --step-index 0
```

Every decision contains the selected model, predicted success by model,
estimated cost, rejected candidates, policy-gate status, and artifact hash.

## Inputs

A model card describes cost and hard constraints:

```json
{
  "name": "provider/model-id",
  "provider": "provider",
  "input_cost_per_million_usd": 0.5,
  "output_cost_per_million_usd": 2.0,
  "context_tokens": 131072,
  "capabilities": ["text", "json", "tools"]
}
```

Training outcomes are JSON Lines:

```json
{"request_id":"train-1","text":"Summarize this.","model":"provider/model-id","acceptable":true,"split":"train","cost_usd":0.002,"group":"summarization"}
```

Every request must have one row for every model. Use `train` rows to fit the
classifier and `calibration` rows to choose the policy. Keep test rows in a
separate file for `semantic-evaluate`.

## How selection works

The classifier estimates each eligible model’s chance of producing an
acceptable result. The policy then compares quality and cost.

| Mode | Selection |
|---|---|
| `quality` | Highest predicted success |
| `balanced` | Predicted success minus a calibrated cost penalty |
| `cost` | Cheapest model close enough to the best prediction |

Capability, provider, context, and request-cost limits run first. Only an exact
classifier and cost setting that passed its calibration gate routes by
default. Other settings fall back to the best fixed calibration model. If that
fallback is ineligible, routing stops instead of making an unsupported choice.

Artifacts are plain JSON and content-hashed. They contain the model cards,
classifier weights, calibrators, evaluated policy settings, fallback model,
training seed, and quality–cost curve.

## Python

```python
import json
from budget_router import SemanticRouter

with open("outputs/router.json", encoding="utf-8") as stream:
    router = SemanticRouter(json.load(stream))

decision = router.route(
    "Extract the invoice fields as JSON.",
    mode="balanced",
    required_capabilities={"json"},
    input_tokens=1200,
    output_tokens=300,
    max_cost_usd=0.05,
)

print(decision.selected_model)
```

For an agent, route the visible conversation before each model call:

```python
import json
from budget_router import SemanticAgentRouter

with open("outputs/router.json", encoding="utf-8") as stream:
    router = SemanticAgentRouter(json.load(stream))

messages = [
    {"role": "user", "content": "Fix the parser regression and run its test."}
]

decision = router.select(
    messages,
    required_capabilities={"tools"},
    output_tokens=1000,
)
print(decision.selected_model)
```

The agent adapter uses only visible messages and tool results. It tracks model
switches but does not execute models or tools.

## OpenAI-compatible proxy

Map artifact models to OpenAI-compatible upstreams in
`examples/proxy.example.json`. Credentials are named by environment variable;
they are never stored in the config.

```bash
budget-router serve-proxy \
  --config examples/proxy.example.json \
  --host 127.0.0.1 \
  --port 8000
```

The gateway exposes `/health`, `/v1/router/decision`, and
`/v1/chat/completions`. Model selection remains separate from provider health,
retry, region, cache, and failover.

## Evidence

The experiments that produced this package used official SWE-bench grading
and fixed model baselines. The evaluated agent router chose between GPT-OSS
20B and Qwen3.6 35B at each model call.

| Evaluation | Router | Fixed Qwen3.6 35B | Cost change |
|---|---:|---:|---:|
| Predeclared novel-repository tasks | 22/50 | 22/50 | −20.80% |
| Full operational set | 26/60 | 24/60 | −20.79% |

The first row’s paired 95% quality interval was −10 to +10 percentage points;
its cost-saving interval was 11.34% to 30.13%. These results show lower cost at
the observed pass rate on one coding-agent workload, not universal model
superiority. Exact aggregate records are in `results/results.json`.

## Evaluate before deployment

1. Run every candidate model on the same representative requests.
2. Split by request or user before training.
3. Train once, select on calibration, and freeze the artifact.
4. Compare the router with every fixed model on untouched outcomes.
5. Shadow live decisions before execution.
6. Monitor quality, cost, route share, and unsupported inputs.
7. Retrain and re-evaluate when models, prices, or traffic change.

## Development

```bash
pip install -e ".[test]"
python -m ruff check src tests
python -m pytest
python -m build
```

Tests require no credentials, network access, Docker, or paid inference.
Report security issues through
[GitHub private advisories](https://github.com/pranavdulepet/budget-router/security/advisories/new).

MIT licensed.
