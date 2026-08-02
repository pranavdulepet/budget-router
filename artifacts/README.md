# Public artifacts

This directory contains frozen protocol metadata, model-free task manifests,
trained router artifacts that are safe to publish, and sanitized aggregate
results. It excludes credentials, raw trajectories, hidden reasoning, patches,
grader logs, terminal output, and benchmark records with restricted
redistribution terms.

## Active protocol

- `active_router_protocol.json` and `.sha256`: machine-readable lock and hash.
- `active_router_protocol_amendment_001_*.json` through
  `active_router_protocol_amendment_012_*.json`: ordered changes and gates.
- `active_router_task_manifest.json`,
  `agent_step_followup_v2_task_manifest.json`, and
  `quality_cost_v1_task_manifest.json`: frozen task identities without gold
  patches or grader tests.

The Markdown protocol and amendments are under `docs/`.

## Prompt-routing study

- `public_router_v1_results.json`: sanitized in-distribution, ArenaHard, and
  ARC-AGI aggregate results with evidence labels and hashes.
- `public_router_v1_quality_cost_curves.json`: all frozen aggregate
  quality-cost curve points used by the article figure.

The full benchmark-derived matrices remain local under `outputs/` and are not
part of the source distribution.

## Coding-agent studies

- `pilot_coding_v6_swebench_grades.json`: compatibility pilot aggregates.
- `router_baseline_v2_results.json` and
  `router_baseline_v2_task_router.json`: static task router.
- `sequential_cascade_v1_results.json`: same-workspace cheap-scout cascade.
- `isolated_stage_v1_results.json` and `isolated_stage_v1_task_gate.json`:
  isolated-scout policy and frozen gate.
- `agent_step_router_v1_results.json`: first per-call study, including the
  zero-cheap-call mechanism failure.
- `agent_step_router_v2_results.json`: guarded 60-task follow-up, official
  grades, paired uncertainty, cost accounting, and route audit.
- `agent_step_router_v2_grading_recovery.json`: grader-recovery bookkeeping.

## Strong-arm qualification

- `quality_cost_v1_compatibility_result.json`: Qwen3.5 397B qualification.
- `kimi128_strong_qualification_result.json`: frozen explicit-submission
  result for Kimi K2.6 128K.
- `kimi128_terminal_workspace_audit.json`: separately labeled post-hoc
  recoverable-workspace diagnostic.

The formal Kimi result remains 4/12 explicit submissions resolved. The
post-hoc terminal-workspace diagnostic is 6/12, tied with Qwen on the same
solve set. These estimands must not be merged.

## Publication rule

Before adding a file here:

1. reduce it to the minimum reproducibility record;
2. remove raw prompts where licensing is unclear;
3. remove patches, logs, terminal output, and hidden reasoning;
4. scan for provider credentials and credential-shaped strings;
5. record source, protocol, and artifact hashes;
6. state whether a result is confirmatory, secondary, or post-hoc.

Never publish `.env` files, provider keys, raw task containers, or local
`outputs/`, `logs/`, and `data/` trees.
