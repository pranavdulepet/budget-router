# Amendment 010: task-block Docker image lifecycle

Status: **prospective infrastructure freeze before Amendment 009 matrix
collection**

Effective date: 2026-07-30

Parent study: `open-router-study-2026-07-29-v1`

## Reason

Amendment 004 required every pending SWE-bench image to be present before a
paid stage began. That was safe for small screens, but the Amendment 009
development and held-out cohorts contain 60 and 100 distinct task images.
Keeping all of those images simultaneously exceeds the available local disk
capacity.

This is an infrastructure-only amendment. It does not change a model, task,
prompt, seed, renderer, cap, sampling setting, treatment, policy, metric,
grader, gate, or budget.

## Superseding lifecycle

For the Amendment 009 fixed development matrix:

1. process task IDs in their frozen manifest order;
2. before any paid call for a task, inspect or pull that task's exact pinned
   `swebench/sweb.eval.x86_64` image;
3. deterministically shuffle the three frozen fixed-model roles within the
   task using seed `20260730`;
4. run all pending fixed roles for that task;
5. persist and validate every terminal episode and trajectory before moving
   to the next task;
6. if any provider or unclassified infrastructure failure occurs, stop the
   stage and retain the image for diagnosis;
7. only after the complete task block is terminal and valid, remove that exact
   task image by its resolved image name.

For Amendment 009 held-out collection, use the same lifecycle but run all six
frozen treatments for a task before removing its image. Treatment order is
the deterministic within-task shuffle already specified in Amendment 009.
All 600 held-out episodes must still be terminal before any official held-out
grade is opened.

The runner is resume-safe by the canonical key
`(study_stage, task_id, treatment)`. A resumed task runs only missing
treatments. It may remove an image only after the full frozen block is
terminal.

## Scientific effect

All comparisons remain paired within exactly the same tasks. Task order is
independent of model outcomes. Image removal happens only after all treatments
for that task, so disk lifecycle cannot select one model's cells over
another's. Docker layer caching is infrastructure state, not an inference
input, and provider inference costs continue to use observed model tokens.

The Amendment 009 USD 3,000.00 hard ceiling and every ordered stage cap remain
unchanged.
