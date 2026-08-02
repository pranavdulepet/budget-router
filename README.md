# Budget Router

Budget Router trains a calibrated model selector from your evaluation
outcomes, applies cost and capability constraints, and returns a model ID
before inference.

It supports any model pool with model cards and a complete request × model
outcome matrix. The package includes a CLI, Python API, agent-step adapter,
held-out evaluator, and OpenAI-compatible proxy.

![Budget Router trains a versioned artifact offline, then filters and scores candidate models before passing one selection to a separate provider layer.](docs/assets/router-architecture.svg)

Model selection is separate from provider health, retry, and failover. The
router does not claim zero-data routing: a model without outcomes is excluded
until the artifact is retrained.

## Install

Python 3.12 is the reference runtime.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[ml,proxy]"
```

Install only the core CLI with `pip install -e .`. The core uses a
standard-library classifier. The `ml` extra adds scikit-learn training; the
`proxy` extra adds FastAPI, HTTPX, and Uvicorn.

## Five-minute quickstart

The included example has two candidate models, four train/calibration
requests, and two untouched test requests. The model names are placeholders;
training and routing make no provider calls.

Train an artifact:

```bash
budget-router semantic-train \
  --outcomes examples/outcomes.example.jsonl \
  --model-cards examples/model_cards.example.json \
  --output outputs/semantic_router.json
```

Route one request:

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

Compare every routed mode with every fixed model on untouched outcomes:

```bash
budget-router semantic-evaluate \
  --artifact outputs/semantic_router.json \
  --outcomes examples/heldout.example.jsonl \
  --output outputs/semantic_evaluation.json
```

Route a visible agent trajectory with the same artifact:

```bash
budget-router agent-route \
  --artifact outputs/semantic_router.json \
  --messages examples/agent_messages.example.json \
  --step-index 0 \
  --mode balanced
```

Each command emits the artifact hash. Routing output includes every
candidate’s predicted success, expected cost, utility, rejection reasons, and
the selected model. It also reports whether that exact policy passed its
calibration gate, the scope of that gate, and whether a fixed fallback was
applied.

## How it works

For request or agent state \(x\) and model \(m\), the classifier estimates

\[
\hat p_m(x)=P(\text{acceptable outcome}\mid x,m).
\]

The default trainer fits hashed text features and model-specific heads,
calibrates each model on a separate calibration split, and evaluates a frozen
cost-weight grid. The balanced policy selects

\[
\arg\max_m\left[
\hat p_m(x)-\lambda
\frac{\mathbb E[C_m]}{\max_j \mathbb E[C_j]}
\right].
\]

Hard eligibility constraints run before scoring.

| Mode | Selection rule |
|---|---|
| `quality` | Highest predicted success among eligible models |
| `balanced` | Predicted success minus normalized cost penalty |
| `cost` | Cheapest model within a configured quality distance of the best; ungated unless explicitly evaluated elsewhere |

Training uses a one-sided calibration non-inferiority gate. Only an exact
classifier-method and cost-weight candidate that passed calibration may route
by default. An unevaluated mode or override—including `cost` mode—falls back
to the best fixed calibration model. If constraints exclude that fallback,
routing fails closed. `allow_ungated_policy=True` is an explicit opt-in for
experimentation. The artifact records the candidate gates, fallback,
calibration metrics, cost curve, training backend, random seed, and content
hash.

## Input contracts

### Model cards

`model_cards.json` is a list, or an object with a `models` list:

```json
{
  "models": [
    {
      "name": "provider/model-id",
      "provider": "provider",
      "input_cost_per_million_usd": 0.5,
      "output_cost_per_million_usd": 2.0,
      "context_tokens": 131072,
      "capabilities": ["text", "json", "tools"],
      "metadata": {"tier": "medium"}
    }
  ]
}
```

Use either token prices or `expected_cost_usd`. Token prices take precedence
when a request supplies token estimates.

### Training outcomes

Training input is JSON Lines. Every request must have one row for every model:

```json
{"request_id":"train-1","text":"Summarize this.","model":"provider/model-id","acceptable":true,"split":"train","cost_usd":0.002,"group":"summarization"}
```

Required fields:

- `request_id`, `text`, and `model`;
- `acceptable`, or `score` plus an optional `acceptable_threshold`;
- `split`, equal to `train` or `calibration`.

`cost_usd` is optional when the model card supplies an expected cost.
`group` enables macro-group calibration so a large dataset does not dominate
selection.

Keep test rows in a separate file. `semantic-evaluate` uses the same schema
without `split`. Add `input_tokens` and `output_tokens` to each request when
model cards use token prices, or supply global estimates on the command line.
Training and evaluation use different input types, preventing held-out labels
from entering the trainer accidentally.

## Python API

### Prompt routing

```python
import json

from budget_router import SemanticRouter

with open("outputs/semantic_router.json", encoding="utf-8") as stream:
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
print(decision.predicted_success)
print(decision.rejected)
```

`SemanticRouter` verifies the artifact hash on load and selects locally. It
does not import a provider SDK or execute the chosen model.

### Agent-step routing

```python
import json

from budget_router import SemanticAgentRouter

with open("outputs/semantic_router.json", encoding="utf-8") as stream:
    agent_router = SemanticAgentRouter(json.load(stream))

messages = [
    {"role": "system", "content": "You are a coding agent with a shell tool."},
    {"role": "user", "content": "Fix the parser regression and run its test."},
]

decision = agent_router.select(
    messages,
    step_index=0,
    mode="balanced",
    required_capabilities={"tools"},
)

print(decision.selected_model)
print(decision.prefix_summary)
print(agent_router.route_counts)
```

The adapter renders only the visible message and tool trajectory. It records
route counts and switches. Your harness remains responsible for executing the
model and appending the result before the next decision.

The repository also contains the fixed two-tier `GuardedAgentStepRouter` used
in the SWE-bench study. It is an experiment-specific policy, not the generic
arbitrary-pool API.

## OpenAI-compatible proxy

Edit `examples/proxy.example.json` so each artifact model maps to an
OpenAI-compatible upstream. Each upstream references a credential by
environment-variable name; configs never contain secret values:

```json
{
  "artifact": "../outputs/semantic_router.json",
  "mode": "balanced",
  "router_model_name": "auto",
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

The upstream map must contain every model in the artifact. Start the proxy:

```bash
budget-router serve-proxy \
  --config examples/proxy.example.json \
  --host 127.0.0.1 \
  --port 8000
```

Inspect a decision without calling an upstream:

```bash
curl -s http://127.0.0.1:8000/v1/router/decision \
  -H 'content-type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarize this text."}],
    "router": {"mode": "balanced", "max_cost_usd": 0.05}
  }'
```

Send normal chat-completions requests to `/v1/chat/completions`. Response
headers expose the selected model, predicted success, and artifact hash.
Streaming is passed through.

The proxy does not implement provider-health learning or automatic endpoint
failover. Put those below semantic selection, or extend the adapter for your
runtime.

## Evaluate before deployment

Do not select a policy on its final test set.

1. Evaluate every model on the same representative requests.
2. Split by request or user before training.
3. Train on `train`; select thresholds on `calibration`.
4. Freeze the artifact.
5. Run `semantic-evaluate` once on a separate complete matrix.
6. Compare routed quality, total cost, route share, and every fixed baseline.
7. Shadow decisions on live traffic before allowing execution.
8. Fall back on unsupported inputs and monitor calibration, cost, and route
   share over time.

By default, the evaluator runs the `quality` and `balanced` policies. It
reports fixed and routed success, total and mean cost, group-macro success,
route counts, exact per-request decisions, input and artifact hashes, and
paired bootstrap intervals against the best fixed model. Evaluating `cost`
mode requires an explicit `--mode cost --allow-ungated-policy`; label that
result as ungated. Application studies should add repeated generations when
outputs are stochastic.

## Research results

The package grew from a classifier-centered routing study.

| Study | Router | Comparator | Observed quality | Cost |
|---|---|---|---:|---:|
| ID prompts, secondary balanced point | Shared classifier | Fixed GPT-5 | 73.64% vs 73.52% | $38.31 vs $56.52 |
| Agent steps, 60-task operational result | Guarded binary router | Fixed Qwen3.6 35B | 26/60 vs 24/60 | $24.54 vs $30.98 |
| Agent steps, predeclared novel-repository result | Same guarded router | Fixed Qwen3.6 35B | 22/50 vs 22/50 | $20.61 vs $26.02 |
| Prompt-router support-boundary check on ARC-AGI | Same frozen artifact | Fixed GPT-5 | 25.25% vs 52.00% | $1.01 vs $52.60 |

Prompt-study costs use the benchmark’s frozen cost field. Agent-study costs
use conservative uncached token accounting. Costs are comparable within each
row, not across the two studies.

The balanced prompt point is secondary evidence; the prespecified prompt
primary did not establish parity with GPT-5. In the agent study, the paired
cost interval excluded zero saving while the quality interval included losses
and gains. Read the [article](docs/technical_blog.md) for the complete design
path, evidence labels, and deployment boundaries.

Key records:

- [active protocol](docs/active_router_protocol.md);
- [public prompt results](artifacts/public_router_v1_results.json);
- [agent-step v1 results](artifacts/agent_step_router_v1_results.json);
- [guarded agent-step v2 results](artifacts/agent_step_router_v2_results.json);
- [agent-step v2 report](docs/agent_step_router_v2_final_report.md);
- [production guide](docs/open_source_router.md);
- [documentation index](docs/README.md).

The `outputs/`, `logs/`, and benchmark `data/` directories are local research
state and are excluded from source distributions. Sanitized aggregate
artifacts under `artifacts/` are the public record.

## Development

```bash
pip install -e ".[test]"
python -m pytest
python -m pytest -m "not local_research"
python -m budget_router --help
python scripts/generate_blog_figures.py
python -m build
```

The complete local suite contains 212 tests. Ten tests carry the
`local_research` marker because they validate ignored benchmark data, frozen
run outputs, or other private study state. Hosted CI runs the remaining 202
fresh-clone tests, then exercises the quickstart and clean wheel separately.

Build and inspect the distributions:

```bash
python -m build
tar -tzf dist/budget_router-0.1.0.tar.gz
```

CI runs the fresh-clone test suite, the self-contained quickstart, a clean
wheel import, and a source-archive check that rejects raw logs, patches, test
output, and local run directories.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[release checklist](docs/release_checklist.md).

## License

MIT. See [LICENSE](LICENSE).
