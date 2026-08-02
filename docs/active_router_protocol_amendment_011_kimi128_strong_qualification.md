# Amendment 011: Kimi K2.6 128K strong-arm qualification

Status: **completed; qualification failed; Tinker strong-candidate search stopped**

Effective date: 2026-07-30

Parent study: `open-router-study-2026-07-29-v1`

## Purpose and relationship to Amendment 009

Amendment 009 stopped correctly after Qwen3.5 397B resolved 1/6 official
compatibility tasks, below its frozen 2/6 threshold. That result and stop
decision are unchanged.

The user subsequently asked whether the new Kimi model should count as a
frontier-like Tinker arm and explicitly authorized completing the study with
the supplied environment-only Tinker credential. This amendment opens a new,
separately labeled development qualification. It is not a post-hoc
replacement inside Amendment 009.

Moonshot reports strong agentic coding results for Kimi K2.6, while Tinker
categorizes it as Large. Our prior same-harness evidence used the 32K Tinker
treatment with a $0.90 episode cap. That treatment resolved 3/12 pilot tasks
and 4/20 held-out tasks, below Qwen3.6 35B. Those outcomes do not establish
whether the 128K variant with a quality-first budget is a strong treatment.

## Frozen candidate

| Field | Value |
|---|---|
| Role | strong candidate |
| Tinker model | `moonshotai/Kimi-K2.6:peft:131072` |
| Tinker size | Large |
| Context | 131,072 tokens |
| Renderer | `kimi_k26` |
| Reasoning | full |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | -1 |
| Maximum output per turn | 8,000 tokens |
| Maximum agent turns | 75 |
| Episode hard cap | $8.00 |

Uncached list-price accounting uses $5.15 per million input tokens and $12.81
per million output tokens. Cached input is reported but is not used to reduce
the conservative study cost.

## Qualification cohort

Use the 12 exact tasks in `artifacts/pilot_tasks.json`. They are previously
graded development tasks, not Amendment 009 held-out tasks. This reuses the
already-frozen Qwen3.6 35B comparison without buying new medium-model calls.

The prior fixed-Qwen treatment officially resolved 6/12:

- `astropy__astropy-12907`;
- `astropy__astropy-14309`;
- `django__django-11451`;
- `django__django-14765`;
- `matplotlib__matplotlib-24570`;
- `matplotlib__matplotlib-24637`.

These known development outcomes define the comparison before Kimi 128K
inference. They are not used to alter the Kimi prompt or decode settings.

## Ordered gate

1. Run one bounded two-turn synthetic tool-call and tool-result smoke.
2. Run Kimi 128K on all 12 tasks under the frozen agent scaffold.
3. Do not open official Kimi grades until all 12 episode records are terminal.
4. Run the pinned official SWE-bench grader and classify only evidenced
   model-caused patch errors.

The candidate passes only if all conditions hold:

- the synthetic interaction is structurally valid;
- at least 11/12 agent episodes are structurally valid;
- at least 8/12 tasks resolve officially;
- at least one resolved task is not among the six fixed-Qwen resolutions;
- there is no systematic renderer or tool defect;
- there are no provider failures or unclassified infrastructure errors.

The eight-resolution bar deliberately exceeds the existing medium treatment's
6/12 result. It is stricter than merely showing that Kimi is Large by catalog
size.

Failure stops further Tinker strong-candidate search. Success admits this
exact treatment as the prospective strong arm, but paid development-matrix
collection begins only after a versioned full-study continuation freezes its
revised cost projections and execution lock.

## Budget

| Stage | Maximum exposure |
|---|---:|
| Synthetic Kimi 128K smoke | $1.50 |
| 12-task official qualification | $96.00 |
| **Maximum** | **$97.50** |

This amount is charged against Amendment 009 recovery and contingency; the
overall USD 3,000.00 ceiling does not increase. Unused budget remains unspent.

Pre-inference accounting correction: the first smoke preflight made no
provider call because its two-full-context worst-case reservation was
$1.35788544, above the original $1.00 sub-cap. Before any affected inference,
the smoke sub-cap was raised to $1.50 and the qualification maximum to $97.50.
No model, prompt, decode, task, gate, or overall study ceiling changed.

Pre-inference harness correction: the first qualification dry run made no
provider call because the shared episode runner enforces the established
8,000-token maximum output per turn. The candidate configuration was corrected
from 16,000 to 8,000 before collection, preserving the same per-turn allowance
used by the fixed-model studies. Context, episode cap, sampling, tasks, and
gate thresholds did not change.

## Invariants

The public task text, Docker environments, mini-swe-agent version, harness
prompt, submission extraction, terminal record schema, official dataset,
grader commit, and secret-redaction rules remain unchanged. No Amendment 009
held-out task, gold patch, grader test, or held-out outcome is accessed.

## Observed result

All 12 episodes were terminal before any official Kimi grade was opened.
The synthetic smoke passed. The full cohort produced 12/12 structurally valid
terminal records, 6/12 submitted patches, 4/12 official resolutions, no
provider failures, and no unclassified grader errors.

The four resolved tasks were all contained in the fixed medium reference's
six-task resolved set. Kimi therefore produced zero unique resolutions beyond
Qwen3.6 35B. Six episodes ended without a patch after exhausting the
budget-aware call reservation or the step limit.

Episode collection cost $51.20758943 and the smoke cost $0.0028202, for a
total of $51.21040963. Collection used 3.03 episode-hours, with a 50.42-minute
maximum episode and throughput of 3.97 tasks/hour.

The candidate failed both quality gates: 4 resolutions were below the frozen
8/12 minimum, and zero unique resolutions were below the frozen minimum of
one. The precommitted decision is in force:
**stop further Tinker strong-candidate search**. See
`docs/kimi128_strong_qualification_report.md` and
`artifacts/kimi128_strong_qualification_result.json`.

## Subsequent infrastructure audit

Amendment 012 does not change this frozen formal-submission outcome or gate
decision. It records that complete tracked diffs were recoverable from the
visible terminal output of two no-submission episodes and that both diffs
passed the same pinned official grader. The separately labeled
terminal-workspace diagnostic is therefore 6/12, exactly tied with Qwen's
six-task solve set. Public interpretation must distinguish that diagnostic
from the 4/12 primary result and must not claim this cohort proved Kimi less
capable than Qwen.
