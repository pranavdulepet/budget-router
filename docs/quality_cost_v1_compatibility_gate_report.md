# Three-tier quality-cost study: strong-arm compatibility result

The proposed Tinker-only three-tier study stopped before router training
because its frozen strong candidate, `Qwen/Qwen3.5-397B-A17B`, did not pass
the official coding-agent admission gate.

This is a failed **model-and-harness qualification**, not a negative result
about model routing. No Amendment 009 development matrix, classifier fitting,
held-out routing, or held-out grading began.

## Frozen gate and observed result

Amendment 009 required:

- a valid two-turn synthetic tool interaction;
- at least 5/6 structurally valid coding-agent episodes;
- at least 2/6 official SWE-bench resolutions;
- no systematic renderer defect, provider failure, or unclassified
  infrastructure error.

| Check | Required | Observed | Pass |
|---|---:|---:|---:|
| Synthetic tool protocol | valid | valid | yes |
| Structurally valid episodes | at least 5/6 | 6/6 | yes |
| Submitted patches | descriptive | 2/6 | — |
| Official resolutions | at least 2/6 | 1/6 | **no** |
| Provider failures | 0 | 0 | yes |
| Unclassified grader errors | 0 | 0 | yes |

The six agent episodes cost a conservative **$23.1296895**; the synthetic
smoke cost **$0.0039105**. Four episodes exhausted their $5 cap without a
submission. Of the two submissions, `pytest-dev__pytest-5631` resolved
officially. The other output for `pytest-dev__pytest-7205` was a Python source
file rather than a unified diff, so the pinned grader classified its patch
application failure as model-caused.

Official grading used dataset SHA-256
`e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076`
and SWE-bench harness commit
`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`.

SWE-bench Verified is retained here for exact comparability with the completed
local studies. Following OpenAI's 2026
[benchmark audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/),
the result is interpreted as a paired compatibility test on a fixed public
workload, not a current frontier-capability estimate.

## Why stopping is the scientific choice

The study's claim required a real cheap/medium/strong outcome matrix. Calling
the 397B model “strong” because of parameter count or a vendor model-card
benchmark would not make it a strong treatment in this exact agent scaffold.
Continuing would spend the development and held-out budget on a baseline that
failed the criterion chosen before its outcomes were visible.

Replacing it after seeing the gate would also violate Amendment 009's explicit
no-result-dependent-replacement rule. A replacement can only be evaluated in
a new, clearly prospective study—not silently relabeled as the original
confirmatory experiment.

The gate did what it was designed to do: it spent about $23 to prevent a much
larger matrix from answering the wrong question.

## What remains valid

The completed guarded two-tier agent-step study remains the project's
strongest same-harness evidence: its frozen router resolved 26/60 tasks for
$24.54, versus fixed Qwen3.6 35B's 24/60 for $30.98 and fixed GPT-OSS 20B's
13/60 for $9.53. It demonstrates genuine per-call cheap-model activation and
cost reduction, although its quality difference is statistically uncertain.

The public 13-model prompt-routing study, ARC-AGI transfer failure, earlier
whole-task experiments, classifier framework, and provider-neutral proxy are
also unchanged.

## Clean continuation options

The intended three-tier question can be restored in one of two prospective
ways:

1. keep Tinker GPT-OSS 20B as cheap and Qwen3.6 35B as medium, and qualify one
   external frontier coding model as strong under the same scaffold; or
2. publish the completed two-tier agent result now and label the Tinker-only
   three-tier attempt as a stopped qualification study.

The first option is the stronger follow-up. It requires a new provider adapter,
an environment-only credential, an explicit external spend ceiling, and a
new compatibility gate before any matrix collection.
