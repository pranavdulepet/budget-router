# Production guide

The [README](../README.md) is the runnable introduction. This guide covers the
data and checks needed before a router controls live traffic.

## Build the outcome matrix

Evaluate every candidate model on the same requests. One missing
request-model outcome changes the training distribution, so Budget Router
rejects incomplete matrices.

Each JSONL row needs:

- `request_id`: the request shared across candidates;
- `text`: only information available when the route is chosen;
- `model`: a name from the model-card file;
- `acceptable`: the task-specific success label;
- `split`: `train` or `calibration`.

`cost_usd` records realized cost. `group` names the unit used for macro
evaluation, such as a product surface, customer cohort, dataset, or
repository.

Choose the split unit that can leak. Split by user for personalized traffic,
conversation for assistants, repository for coding agents, and time for
systems exposed to model or workload drift. Keep test outcomes in a separate
file.

Model cards define opaque model IDs, providers, context limits, capabilities,
and prices. They can describe hosted models, local servers, fine-tunes, or
complete agent endpoints. See
[`examples/model_cards.example.json`](../examples/model_cards.example.json)
and [`examples/outcomes.example.jsonl`](../examples/outcomes.example.jsonl).

## Train, freeze, and test

```bash
budget-router semantic-train \
  --outcomes examples/outcomes.example.jsonl \
  --model-cards examples/model_cards.example.json \
  --output outputs/semantic_router.json

budget-router semantic-evaluate \
  --artifact outputs/semantic_router.json \
  --outcomes examples/heldout.example.jsonl \
  --output outputs/semantic_evaluation.json
```

Training fits independent and shared task-model classifiers, calibrates model
probabilities, and evaluates a cost-weight grid on calibration data. The
artifact stores the chosen policy, model cards, weights, calibrators, input
hashes, and its own content hash.

The default gate compares each candidate policy with the best fixed
calibration model. A policy that misses the configured quality margin cannot
route by default. An unevaluated override falls back to the fixed model. If a
hard constraint excludes that fallback, routing stops.

The test report compares each routed policy with every fixed candidate. It
includes success, group-macro success, total and mean cost, route counts,
per-request choices, and paired bootstrap intervals.

Do not tune the artifact after opening its test report. New models, prompts,
tools, evaluators, prices, or traffic require a new artifact and a new test.

## Route agent calls

`SemanticAgentRouter` applies the same arbitrary-pool artifact before each
model call. It reads only the visible message and tool history.

```python
import json

from budget_router import SemanticAgentRouter

with open("outputs/semantic_router.json", encoding="utf-8") as stream:
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

The decision contains predicted success and cost for each model, rejection
reasons, the artifact hash, a bounded prefix summary, and a switch indicator.
The application executes the selected model and appends its response.

Agent training rows need labels for the exact pool being routed. Group all
prefixes from one task or trajectory into one split. A label learned for one
model pool, prompt, tool protocol, or harness is not portable to another.

The repository retains the two-tier guarded router used in the SWE-bench
study. That policy is research evidence, not the generic public interface.
The final policy calibrated against visible development-run prefixes, added
explicit guards, and used the cheaper model on 12.70% of held-out calls. Its
development showed that activation must be tested dynamically.

## Run the proxy

[`examples/proxy.example.json`](../examples/proxy.example.json) maps artifact
model IDs to OpenAI-compatible upstreams. It stores environment-variable
names, never credential values.

```bash
budget-router serve-proxy \
  --config examples/proxy.example.json \
  --host 127.0.0.1 \
  --port 8000
```

Send chat-completions requests to `/v1/chat/completions`.
`POST /v1/router/decision` returns a decision without calling an upstream.
`GET /health` returns the artifact hash and model pool.

The proxy selects models. It does not learn provider health or perform
automatic endpoint failover. Keep retries, regional availability, deadlines,
and cache-aware provider selection in the execution layer.

## Deployment checks

Before routing live requests:

1. Compare the frozen policy with every fixed candidate on untouched data.
2. Add a fixed fallback for inputs outside the training distribution.
3. Shadow decisions before allowing execution.
4. Log the artifact hash, route, outcome, latency, cost, and provider errors.
5. Monitor calibration, route share, input drift, and budgets separately.
6. Preserve session consistency unless step-level switching was evaluated.
7. Revalidate after any model, prompt, harness, evaluator, price, or workload
   change.

The package is provider-neutral. Its evidence is workload-specific. The
[article](technical_blog.md) reports both the successful held-out cost result
and the distribution-shift boundary that constrains its use.
