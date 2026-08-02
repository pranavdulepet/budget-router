# Contributing

Budget Router welcomes changes that improve hard-budget accounting, routing
policies, replay, calibration, experiment reproducibility, documentation, and
provider-neutral integrations.

## Development setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
budget-router --help
```

Provider and coding-runner dependencies are optional:

```bash
python -m pip install -e ".[tinker,runner]"
```

Tests submitted to the project must not require paid API calls, provider
credentials, Docker, or network access. Use deterministic fixtures for core
router behavior.

## Pull requests

- Keep the core SDK provider-neutral.
- Add tests for behavior changes.
- Preserve exact-decimal hard-cap checks and worst-case reservations.
- Keep private reasoning out of traces and handoffs.
- State the model, renderer, prompt, harness, sampling settings, and price
  snapshot for any empirical claim.
- Label training-only or exploratory evidence explicitly.

Do not commit `.env` files, API keys, raw provider logs, raw model reasoning,
task workspaces, or raw terminal-grader output. The `outputs/` and `traces/`
directories are ignored intentionally.

No contributor should spend money on behalf of the project without an explicit
budget and authorization outside the pull request.
