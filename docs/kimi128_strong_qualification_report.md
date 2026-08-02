# Kimi K2.6 128K strong-arm qualification result

The prospective Kimi K2.6 128K qualification failed its frozen admission gate.
The exact Tinker treatment is therefore **not** admitted as the strong arm, and
the project stops further Tinker strong-candidate search as required by
Amendment 011.

This is a model-and-agent-treatment qualification result. It is not a claim
that Kimi K2.6 is globally weak, and it is not a negative evaluation of model
routing: no new three-tier router was trained or tested.

A subsequent infrastructure audit recovered two complete tracked diffs from
the visible terminal output of no-submission episodes. Both passed the same
pinned official grader. The frozen formal-submission primary remains 4/12,
but the separately labeled terminal-workspace diagnostic is 6/12—exactly tied
with Qwen's solve set. This correction changes the interpretation, not the
precommitted gate decision.

## Frozen gate and official result

The candidate was `moonshotai/Kimi-K2.6:peft:131072` with 128K context, full
reasoning, up to 75 agent turns, 8,000 output tokens per turn, and an $8 hard
cap per task. It ran once on the 12 exact public development tasks frozen in
`artifacts/pilot_tasks.json`. Official grades remained sealed until all 12
episodes were terminal.

| Check | Required | Observed | Pass |
|---|---:|---:|---:|
| Synthetic tool protocol | valid | valid | yes |
| Structurally valid episodes | at least 11/12 | 12/12 | yes |
| Submitted patches | descriptive | 6/12 | — |
| Official resolutions | at least 8/12 | 4/12 | **no** |
| Recoverable terminal-workspace resolutions | at least 8/12 | 6/12 | **no** |
| Resolutions beyond fixed Qwen medium | at least 1 | 0 | **no** |
| Provider failures | 0 | 0 | yes |
| Unclassified grader errors | 0 | 0 | yes |

The four official resolutions were:

- `astropy__astropy-14309`;
- `django__django-11451`;
- `django__django-14765`;
- `matplotlib__matplotlib-24570`.

All four were already among the six tasks resolved by the fixed
Qwen3.6 35B-A3B reference. The two apparent Qwen-only wins were:

- `astropy__astropy-12907`;
- `matplotlib__matplotlib-24637`.

The Kimi trajectories for those tasks ended without an explicit submission,
but their final visible tool outputs contained complete tracked diffs. Both
recovered diffs passed the official grader. The combined recoverable Kimi
solve set therefore equals Qwen's six-task solve set exactly and adds no
unique solve.

## What happened operationally

Kimi's explicit submissions were often good: 4 of its 6 submitted patches
passed the official grader. The original harness nevertheless understated
recoverable completed work because explicit submission was a single point of
failure.

Six episodes exhausted either the budget-aware call reservation or the
75-step agent limit without a formal patch. Five stopped when their remaining
budget fell below the conservative reservation for another maximum-size
model call; one reached exactly 75 calls. Two of the six had complete,
officially correct work in visible terminal output. The other four historical
workspaces had already been destroyed, so their final tracked state cannot be
reliably reconstructed or graded.

Amendment 012 repairs this prospectively: every ephemeral repository runner
now captures `git diff --binary --no-ext-diff HEAD --` after any terminal path
and before cleanup, persists it as a separate sidecar, and records its hash
and capture status. The explicit submission remains unchanged and primary;
capture failure cannot mask the original episode result.

| Operational metric | Observed |
|---|---:|
| Episode cost | $51.2076 |
| Synthetic smoke cost | $0.0028 |
| Mean episode cost | $4.2673 |
| Median episode cost | $4.0659 |
| Total episode wall time | 3.03 hours |
| Mean episode wall time | 15.13 minutes |
| Median episode wall time | 7.53 minutes |
| Maximum episode wall time | 50.42 minutes |
| Throughput | 3.97 tasks/hour |

For comparison, fixed Qwen cost $6.1971 and took 3,300 seconds on the same
cohort. Kimi therefore cost 8.26 times as much and took 3.30 times as long.

The full qualification used $51.2104 including the smoke, versus its $97.50
maximum. Unused authorization was not spent.

## Reproducibility

Official grading used:

- SWE-bench Verified dataset SHA-256
  `e0afd44653d31d3da6adc5b04bb3af38800cd9aec13af6c21bd93d012d008076`;
- SWE-bench harness commit
  `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`;
- prediction SHA-256
  `67e8dad41691d7b3b2b15560b0e33ee2795bfe3cd49ac14dda654babcc52ae06`;
- official report SHA-256
  `110cb5ce6cd0c5c9ce29fba6575928c68c2d634da7b937268fe55fdd9ac3fc58`.

The post-hoc diagnostic used the same dataset and harness. Its two-patch
prediction SHA-256 is
`d1bb0a18441d869b83eb780b097a979b2e810462ce58142d1b880f13f8efddbb`;
both predictions resolved. The sanitized audit, including source trajectory
and recovered-patch hashes, is
`artifacts/kimi128_terminal_workspace_audit.json`.

The sanitized machine-readable result is
`artifacts/kimi128_strong_qualification_result.json`. Raw local trajectories,
episode records, predictions, and official reports are under
`outputs/quality_cost_v2/kimi128_gate/`.

As with the earlier compatibility result, SWE-bench Verified is used as a
fixed paired workload, not as a current frontier leaderboard claim.

## Consequence for the project

The Tinker catalog did not yield an admitted strong arm under either
prospective gate:

- Qwen3.5 397B resolved 1/6 in Amendment 009;
- Kimi K2.6 128K formally resolved 4/12 in Amendment 011; the Amendment 012
  recoverable-workspace diagnostic ties fixed Qwen medium at 6/12 with no
  unique solve, while costing 8.26 times as much.

The scientifically valid conclusion is to stop Tinker model shopping. The
completed guarded two-tier agent-step router remains the positive agent-level
result: 26/60 resolutions for $24.54 versus fixed Qwen's 24/60 for $30.98,
with genuine cheap-model activation but statistically uncertain quality
improvement.

A future three-tier study needs one prospectively qualified external frontier
coding treatment or a newer benchmark/provider combination. It must be a new
versioned study, not a post-hoc replacement in the completed Tinker gate.
