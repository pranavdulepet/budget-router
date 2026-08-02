# Protocol amendment 003: provider-neutral context budgeting

Status: **normative infrastructure repair during development screening**

Effective 2026-07-29 under the active protocol's mandatory infrastructure stop
and repair rule. This amendment does not alter the research question, task
split, model pool, system/task prompts, seeds, output allowance, episode caps,
selection rules, held-out design, or USD ceiling.

## Observed defect

After tool-call recovery let Nemotron Nano make real progress, its accumulated
history plus the fixed 8,000-token output allowance exceeded its 65,536-token
context window. The same failure then occurred for GPT-OSS 20B at its
32,768-token context. This established a shared harness defect rather than a
model-specific quality result.

The active collection queue was stopped immediately. Four in-flight episodes
had no terminal records. Their full per-episode caps, totaling $2.20, are
charged to a conservative interrupted-exposure ledger even though actual
provider billing may be lower. They produce no quality outcomes and are
recollected with their original treatment settings.

The terminal GPT-OSS 20B provider-error episode on
`astropy__astropy-7671` remains in the cost ledger but is invalidated for
quality and structural metrics. It is recollected with the same task, seed,
prompt, cap, and model settings.

## Repair

Before every Tinker request, render the actual prompt and reserve the complete
locked output allowance inside that treatment's declared context window. If
the prompt would exceed the window:

1. preserve the system prompt and original public task request;
2. preserve the newest complete assistant/observation exchanges;
3. remove the oldest complete exchanges until the full output allowance fits;
4. never leave an orphan tool response;
5. record the number of omitted history messages in provider metadata.

The repository workspace remains persistent, so the model can re-inspect files
after old transcript turns leave the sliding window. If the two preserved
anchors and newest exchange still cannot fit, fail locally before a provider
call.

This is a generic harness invariant applied identically using each model's
predeclared context size. It is not prompt, cap, or model-quality tuning.

## Recollection

After unit and regression tests plus protocol smokes at both 32K and 64K
contexts, recollect only invalid or interrupted task/model pairs using their
original seeds and treatments. Preserve and count every diagnostic or
interrupted cost. No official screen grade may be opened until collection is
again terminal.

