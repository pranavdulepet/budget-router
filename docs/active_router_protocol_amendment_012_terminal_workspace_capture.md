# Amendment 012: terminal workspace patch capture and Kimi audit

Status: **normative infrastructure repair after Amendment 011**

Effective date: 2026-07-30

Parent study: `open-router-study-2026-07-29-v1`

## Reason

Amendment 011 correctly recorded the explicit mini-swe-agent submission
outcome: Kimi K2.6 128K submitted 6/12 patches and resolved 4/12 tasks. After
the user questioned whether this understated Kimi, a read-only trajectory
audit found two no-submission episodes whose visible final tool output
contained a complete tracked source diff:

- `astropy__astropy-12907` ended at its budget reservation boundary
  immediately after `git diff`;
- `matplotlib__matplotlib-24637` reached the 75-step limit immediately after
  creating and inspecting `patch.txt`.

The runner cleaned up each Docker container without independently capturing
the terminal workspace. This made the explicit model submission command a
single point of failure and conflated completed coding work with
terminalization behavior.

This is a shared evaluation-harness defect. It does not authorize paid
inference, alter any model treatment, or change a frozen primary outcome.

## Post-hoc diagnostic

The two complete diffs were recovered verbatim from the saved visible tool
outputs and graded with the same pinned official SWE-bench harness and
dataset. Both resolved:

- `astropy__astropy-12907`;
- `matplotlib__matplotlib-24637`.

The Amendment 011 primary score remains **4/12**, because those patches were
not formal submissions under the frozen protocol. The separately labeled
terminal-workspace diagnostic is **6/12**. Its six resolved IDs are exactly
the fixed Qwen3.6 35B reference's six resolved IDs.

This changes the interpretation, not the gate decision:

- the primary formal-submission comparison is Kimi 4/12 versus Qwen 6/12;
- recoverable completed work ties at 6/12;
- Kimi still falls below the frozen 8/12 admission threshold;
- Kimi still adds zero resolution beyond Qwen;
- Kimi costs $51.20758943 versus Qwen's $6.197096175 on the cohort and has
  materially lower throughput.

The Amendment 011 decision to stop further Tinker strong-candidate search
therefore remains in force. The project must not claim that the 128K Kimi
treatment was proven less capable than Qwen on these 12 tasks. It may claim
that Kimi was not admitted as a quality-cost strong arm.

## Prospective infrastructure repair

Every project runner using an ephemeral repository workspace must, after the
agent reaches any terminal path and before container cleanup:

1. run a harness-owned, zero-model-call tracked workspace diff;
2. use `git diff --binary --no-ext-diff HEAD --`;
3. persist the exact diff separately from the model's explicit submission;
4. record capture attempt, return code, byte count, SHA-256, artifact path,
   and any capture error in the episode record;
5. never replace or silently merge the explicit submission;
6. never let a capture failure mask the agent's original terminal status;
7. retain explicit and captured patches when they differ.

The automatic capture includes tracked workspace modifications only.
Untracked helper, reproduction, test, and patch files are deliberately not
promoted into a grader prediction. A future protocol may add an independently
specified untracked-file policy before collection.

For historical studies, explicit submission remains the primary estimand.
Recovered workspace patches are post-hoc diagnostics only. A future study may
make terminal workspace patches primary only by declaring that rule before
any affected model inference.

## Scope and budget

This amendment adds no task, seed, model, outcome recollection, or provider
call. It changes no held-out label, gate, threshold, cost cap, or USD ceiling.
The post-hoc official grading uses two already-generated development diffs and
incurs zero model cost.

The sanitized diagnostic is
`artifacts/kimi128_terminal_workspace_audit.json`.

