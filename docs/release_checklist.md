# Version 0.1 release checklist

## Complete locally

- Apache-2.0 license, contribution guide, and security policy.
- Python 3.12 package with CLI, Python API, arbitrary-pool semantic router,
  agent-step adapter, held-out evaluator, and OpenAI-compatible proxy.
- Calibration-gate fallback: only exact passing policy candidates route by
  default. Failed or unevaluated policies use the best fixed calibration
  model and fail closed if constraints exclude it.
- Complete self-contained example for training, prompt routing, held-out
  evaluation, and agent-step routing.
- Frozen protocols through Amendment 012 and sanitized result artifacts for
  the prompt, static task, cascade, isolated scout, agent-step v1, guarded
  agent-step v2, three-tier compatibility, and Kimi qualification studies.
- Article, four deterministic SVG figures, Medium-ready PNG exports,
  aggregate figure data, README, and production guide.
- CI jobs for the 201-test fresh-clone suite, documented quickstart, package
  build, archive privacy check, and clean wheel import.
- Local full suite: 211 tests passed.
- Local wheel and source archive rebuilt from current source.
- Source archive excludes top-level `data/`, `logs/`, and `outputs/`, plus raw
  patches, run logs, and grader output.
- Clean Python 3.12 wheel installation and CLI/API smoke check.
- Dedicated public repository:
  `https://github.com/pranavdulepet/budget-router`.
- Repository, documentation, issue, release, citation, contribution, and
  private vulnerability-reporting metadata.

## Required before publication

1. Run the hosted CI workflow from the first fresh public clone.
2. Review the article under Pranav Dulepet’s byline.
3. Tag `v0.1.0` and attach the wheel, source distribution, sanitized result
   artifacts, and article. PyPI publication is optional.

## Research boundaries

- Do not retune the public prompt artifact on ID test, ArenaHard, or ARC-AGI
  labels and report against those same labels.
- Do not retune agent-step v1 on its 20 terminal outcomes.
- Do not retune guarded agent-step v2 on its 60 official outcomes.
- Keep Kimi’s 4/12 frozen explicit-submission result separate from its 6/12
  post-hoc terminal-workspace diagnostic.
- A future benchmark, model pool, task split, or paid study requires a
  prospective protocol amendment.
- The next agent study should use a new repository- or time-disjoint cohort,
  representative live-prefix activation, a support/abstention rule, complete
  fixed baselines, and multiple trajectory seeds.
