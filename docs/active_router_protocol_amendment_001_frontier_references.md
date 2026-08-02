# Protocol amendment 001: reusable frontier-model references

Status: **normative, pre-paid-collection**

Authorized by the user on 2026-07-29.

This amendment augments `docs/active_router_protocol.md`. It does not alter the
primary hypothesis, candidate gates, held-out split, or incremental Tinker
ceiling.

## Purpose

The final report and blog should determine whether the learned router also
compares favorably with frontier fixed-model baselines on the evaluated
benchmarks. Existing public frontier-model runs should be reused when doing so
is scientifically valid; they should not be rerun merely to populate a table.

## Public reuse order

Use these sources in order:

1. LLMRouterBench standardized per-instance outputs and costs for its flagship
   model pool.
2. Public SWE-bench or SWE-Router per-instance predictions with terminal
   official grades.
3. Other official model-provider evaluation artifacts only when exact
   per-instance membership and evaluation conditions are recoverable.

Candidate reference families include contemporary OpenAI, Anthropic, Google,
DeepSeek, Qwen, and other frontier models already present in those public
artifacts.

## Inclusion criteria

A public frontier result may enter the exact-task SWE table only when:

- all compared task IDs can be matched exactly;
- a terminal official grade or reconstructable prediction is available per
  task;
- the dataset version and evaluation harness are documented;
- unresolved and missing predictions remain in the denominator;
- no held-out result is used for training, method selection, threshold tuning,
  or debugging.

If the public run used a different agent, prompt, token allowance, or inference
date, it is labeled an **external quality reference**, not a paired
same-harness baseline. Cost is compared only when token usage, pricing date,
and accounting conventions are compatible.

On LLMRouterBench, existing standardized frontier outputs may be evaluated
directly under that benchmark's frozen splits and accounting because every
router method reads the same recorded outcome matrix.

## Claim rule

The blog may say the router exceeded an external frontier reference only for a
clearly named metric and exact task set. It must state any harness or budget
difference next to the claim.

The primary confirmatory comparison remains the frozen Tinker fixed-model
matrix. Public frontier references are secondary context and cannot rescue a
failed primary result.

## Fresh frontier calls

No fresh OpenAI, Anthropic, Google, or other non-Tinker frontier inference is
authorized by this amendment.

If compatible public per-instance artifacts do not exist and fresh frontier
runs would materially strengthen the conclusion:

1. finish the no-cost reuse audit;
2. calculate the exact number of calls and worst-case USD exposure;
3. propose a same-task, same-harness frozen arm;
4. ask the user for explicit additional-credit authorization;
5. version and lock a second amendment before making any call.

Any such future authorization is separate from the current incremental
Tinker ceiling of $220.

