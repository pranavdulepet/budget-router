# Coding-agent pool pilot: early-stop report

## Outcome

The lower-cost Tinker pilot did not pass its structural gate. It stopped after
15 of 60 planned episodes because every model failed on the first three tasks.
With three failures, a model can finish with at most 9 valid episodes out of
12, below the required 10.

The fixed 500 × 5 × 3 matrix was not started.

## Observed result

| Model | Episodes | Structurally valid | Provider timeouts | Best possible final score |
|---|---:|---:|---:|---:|
| Nemotron-3 Nano | 3 | 0 | 3 | 9/12 |
| Qwen3.6 35B-A3B | 3 | 0 | 3 | 9/12 |
| Nemotron-3 Super | 3 | 0 | 3 | 9/12 |
| Inkling | 3 | 0 | 3 | 9/12 |
| Kimi K2.6 | 3 | 0 | 3 | 9/12 |

Each real-task request used one worker, a 128-token output allowance, and a
180-second request timeout. The persisted prompts were small; the first task's
Nano prompt contained 1,926 rendered input tokens. Every request timed out
before returning a tool call.

The result is an infrastructure or treatment-compatibility failure, not a
coding-quality comparison. No model reached the point where it could inspect
or edit the repository.

## Control evidence

The revised five-model synthetic tool smoke passed for every model. A later
Nano health probe also completed a two-turn tool exchange in 9.5 seconds.
Therefore credentials, tokenizers, renderers, and basic structured tool parsing
work for short synthetic prompts. The failure appears only on real mini-swe
task prompts.

## Cost

The formal early-stop run conservatively cost $0.84408576. Including the
revised smoke, diagnostics, completed exploratory calls, and full reservations
for calls whose usage could not be observed, total conservative exposure under
the $60 authorization is $2.336520245.

## Decision

Do not spend the remaining authorization and do not start the fixed matrix.
First diagnose the Tinker sampling path for real mini-swe prompts. Replacing
individual models is not yet justified because four model families failed in
the same way before tool execution.

## Postmortem

The subsequent diagnosis identified the cause in the harness: it multiplied
the experiment seed by 1,000 before every model call, producing values around
20.3 billion. Tinker SDK 0.23.1 accepted those integers, but the sampling
backend left the requests unresolved. The original 15 episodes are therefore
invalid infrastructure observations and must not be used as model-quality or
compatibility evidence.

The fix removes the multiplication and defensively maps Tinker seeds into the
signed 32-bit range. A post-fix real mini-swe episode completed 29 model calls
and multiple tool round trips with zero provider failures. See
`docs/tinker_prompt_timeout_diagnosis.md` for the controlled evidence.
