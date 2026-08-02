# Protocol amendment 004: sequential Docker image preflight

Status: **normative infrastructure repair during development screening**

Effective 2026-07-29 under the active protocol's mandatory infrastructure stop
and repair rule. This amendment changes no model-facing treatment or research
design.

## Observed defect

Both treatments for `matplotlib__matplotlib-26291` failed before any Tinker
call because its official SWE-bench Docker image was absent. Parallel workers
attempted environment startup at the same time and returned
`CalledProcessError`. The two zero-cost terminal records are invalidated and
recollected with their original settings.

The queue was stopped immediately. Four other episodes were in flight without
terminal records. Their complete caps, totaling $2.60, are added to the
conservative interrupted-exposure ledger; they yield no quality outcomes and
are recollected unchanged.

The first manual image pull then showed that Docker's internal VM disk was
full. Twelve exact SWE-bench images from tasks outside the active 44-task
manifest were removed. Active-protocol images, all source and result files, and
unrelated application images were preserved. Removed images are recoverable
from the official registry.

## Repair

Before constructing a Tinker adapter or launching a paid episode:

1. resolve the official Docker image for every pending locked task;
2. inspect local availability;
3. pull missing images sequentially;
4. stop locally if any pull or registration fails;
5. only after all pending images are present, launch paid workers.

This changes only environment preparation. Tasks, prompts, models, seeds,
caps, context handling, concurrency, and grading are unchanged.

