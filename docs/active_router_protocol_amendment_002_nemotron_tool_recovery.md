# Protocol amendment 002: conservative Nemotron tool-call recovery

Status: **normative infrastructure repair during development screening**

Effective 2026-07-29 under the active protocol's predeclared infrastructure
repair rule. This amendment does not alter the research question, task split,
models, prompts, seeds, episode caps, selection rules, held-out design, or USD
ceiling.

## Observed defect

The first Nemotron Nano development episode used the locked recommended
`nemotron3` renderer and reached Tinker without a provider error. Across 75
calls, the model visibly attempted bash calls, but the cookbook parser returned
zero structured calls. Most responses contained the renderer's documented
Nemotron XML function body without its opening `<tool_call>` wrapper; a smaller
number contained an unambiguous bare JSON bash-call object. The agent therefore
returned a format error on every turn and never executed a command.

This is a model/renderer interoperability defect, not a coding-quality outcome.
The affected episode remains in the cost ledger but is excluded from screen
quality, structural-validity, and latency estimates. Its exact key is recorded
in the machine-readable amendment.

## Repair

The Tinker adapter may recover a call only for a `nemotron3*` renderer when the
cookbook produced no structured tool call and the visible response contains
exactly one unambiguous bash command in one of these observed forms:

1. one `<function=bash><parameter=command>…` XML body, with no second function
   and no suffix other than an optional closing `</tool_call>`; or
2. an exact bare JSON bash-call object, including the observed
   `[Makes a tool call: …]` wrapper.

Ambiguous, multiple, differently named, extra-key, or ordinary-text responses
remain format errors. Recovered calls are tagged in provider metadata, and the
raw malformed markup is removed from the visible assistant content before the
structured call is added to history.

Retrospective validation found that this rule would recover 74 of the 75
malformed calls in the diagnostic episode while rejecting ambiguous examples.

## Recollection and stop rule

After unit tests and a synthetic protocol check, rerun only the invalidated
task/model pair with its original task, seed, prompt, cap, and treatment
settings. Preserve both records and count both costs.

If the repaired episode does not execute structured tools or exposes another
systematic protocol defect, exclude Nemotron Nano and do not run its remaining
screen tasks. If it passes, continue the locked 12-task screen; only the
post-repair episode counts for the invalidated pair.

