# Final report: the paired cost-aware router study

## Outcome

We completed the reduced study, trained a real task-level classifier, froze it
before test collection, and ran all saved patches through the official
SWE-bench terminal harness. The learned router did not beat the best fixed
model. It selected Qwen 3.6 35B-A3B for every held-out task and therefore
matched that baseline exactly: 13/20 tasks resolved for $8.4376.

This is a scientifically useful negative routing result. The held-out task
oracle also resolved 13/20 because every Inkling and Kimi success was already
a Qwen success. No task-only router over this model pool could have improved
quality on the observed test outcomes.

We then completed the separately frozen on-policy sequential arm. A six-call
Qwen 8B scout handed the same workspace and full visible transcript to Qwen
35B under the same $0.90 total cap. The cascade resolved 11/20 for $6.7167:
20.4% lower total cost than fixed Qwen, but two fewer resolutions. It failed
the frozen success rule. A hindsight fixed/cascade selector could retain
13/20 for $5.8810, showing an observed cost-routing opportunity but not a
trained deployable gate.

## Design

The source was SWE-bench Verified at revision
`91aa3ed51b709be6457e12d00300a6a596d4c6a3`. The frozen study manifest
`caec3eab9575b53f9fef2bbb670e8292186b71857798d37575d38c2b20ec156d`
specified 60 training, 20 calibration, and 20 held-out test tasks. Repositories
were disjoint across those splits.

The initial screen compared:

- Qwen/Qwen3-8B;
- Qwen/Qwen3.6-35B-A3B;
- thinkingmachines/Inkling;
- moonshotai/Kimi-K2.6.

Each task/model episode used the same mini-swe-agent workflow and treatment-
specific frozen renderer and sampling settings. Qwen 8B had a $0.35 hard cap;
the other treatments had a $0.90 hard cap. A no-patch episode counted as
unresolved.

The 32-task screen combined a frozen 12-task pilot cohort with 20 new training
tasks. It found two oracle-only gains beyond Qwen, satisfying the
preregistered complementarity gate, but only one task where Qwen 8B was a
successful cheap substitute. The cheap-opportunity gate required three, so
Qwen 8B was pruned. Qwen 35B, Inkling, and Kimi survived.

We then formed a complete outcome matrix for those three models on 60 training
tasks and 20 calibration tasks. The training matrix reused 12 already-graded
pilot tasks and added the screen plus 28 expansion tasks.

## What was trained

This study's trained router is a static model-selection classifier. It is
distinct from the later fixed sequential cascade and from a future learned
stage-policy gate.

For each surviving model, it fits a balanced logistic success head to 1,024
hashed issue-text features: normalized unigram and adjacent-bigram counts plus
a coarse length feature. The input is only the public problem statement.
Repository identity, gold patches, grader tests, grader logs, and test labels
are excluded.

At inference time it computes

\[
s_m(x)=\hat p_m(Y=1\mid x)
       - \lambda\frac{\mathbb E[C_m]}{\max_j \mathbb E[C_j]}.
\]

Learning rate, L2 regularization, and the cost penalty were selected on the
20-task calibration split by resolved count, then observed cost, then Brier
score. The selected penalty was 0.8. It selected Qwen on all 20 calibration
tasks and resolved 11.

The router artifact was frozen before any test model call or test grade:

- artifact hash:
  `e60eeefa2f7428880603421d2fcf68013eb45181159dafb5ad9d907c1bfcb0f6`;
- freeze hash:
  `34ef59db74012923111fe835e76b2d5c0d41cfeefd14769f90249fa780a79a25`.

The evaluator verifies both hashes and rejects an incomplete test matrix.

## Held-out results

The official harness commit was
`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`. It graded 35 submitted
patches with zero final harness errors. Twenty-five no-patch episodes remained
in the denominator as unresolved.

| Policy | Submitted | Resolved / 20 | 95% Wilson interval | Total cost | Cost / resolved |
|---|---:|---:|---:|---:|---:|
| Fixed Qwen 3.6 35B-A3B | 16 | 13 (65%) | 43.3%–81.9% | $8.4376 | $0.6490 |
| Fixed Inkling | 13 | 10 (50%) | 29.9%–70.1% | $10.3939 | $1.0394 |
| Fixed Kimi K2.6 | 6 | 4 (20%) | 8.1%–41.6% | $12.9720 | $3.2430 |
| Frozen router | 16 | 13 (65%) | 43.3%–81.9% | $8.4376 | $0.6490 |
| Hindsight task oracle | — | 13 (65%) | — | — | — |

All ten Inkling successes and all four Kimi successes were subsets of Qwen's
thirteen successes. Paired exact McNemar p-values were 0.25 for Qwen versus
Inkling, 0.0039 for Qwen versus Kimi, and 0.0313 for Inkling versus Kimi.
These are descriptive, unadjusted comparisons on a small test set.

The frozen router's delta from the best fixed model is exactly zero in both
resolution and cost because the action sequence is identical.

## Sequential held-out results

The optional cascade reservation was present in the source study before static
test collection. Exact phase details were frozen before any cascade outcome in
`artifacts/sequential_cascade_study_v1.json`, hash
`3ebd1f7c6d893a2d14174deeb8273a519ede433cb79008bc2287cb72abdfa8f3`.
Because those details were finalized after the static result, this is a
secondary exploratory paired experiment.

| Policy | Submitted | Resolved / 20 | 95% Wilson interval | Total cost | Cost / resolved |
|---|---:|---:|---:|---:|---:|
| Fixed Qwen 35B | 16 | 13 (65%) | 43.3%–81.9% | $8.4376 | $0.6490 |
| Qwen 8B → Qwen 35B | 17 | 11 (55%) | 34.2%–74.2% | $6.7167 | $0.6106 |

All 20 episodes handed off after six scout calls; the scout submitted no task
by itself. Scout work cost $0.1491 and finisher work $6.5676. The official
harness graded every submitted patch with zero errors.

The cascade lost two fixed-Qwen successes and added none. Its paired
resolution delta was -10 points, bootstrap 95% interval [-25, 0] points, with
two-sided exact McNemar \(p=0.5\). Total cost fell by $1.7209, with bootstrap
interval [-$3.8772, $0.4142].

The result is mixed: cheaper execution, lower quality. A post-hoc
quality-preserving oracle selects cascade on 11 tasks and fixed on 9, retaining
13 resolutions for $5.8810, 30.3% below fixed cost. Because that choice uses
terminal labels, it is an upper bound and cannot be deployed or reported as a
trained router.

## Cost

The follow-up collected 296 new valid episodes for a recorded exposure of
$158.708486065. One abandoned provider hang is not in the ledger, so we reserve
the full $0.90 episode cap and report a conservative incremental exposure of
$159.608486065 for the static study. The sequential arm added $6.716709795,
bringing reserve-adjusted current exposure to $166.325195860. This remained
below the $190 working and $200 absolute limits.

The study also reused $38.793438190 of valid pilot observations. Counting those
outcomes, the empirical data used by the static and sequential studies cost
$204.218634050; the reused amount was not charged again during the follow-up.

## Interpretation

The classifier exists and was evaluated correctly. It collapsed to a fixed
policy because the calibration data favored the model that was both more
accurate and cheaper on average. The test oracle confirms that this conservative
choice did not leave a model-routing quality gain on the table.

This does not prove that routing never works. It says that, under this frozen
agent, budget, price snapshot, model pool, task sample, and single stochastic
run per treatment, there was no held-out complementarity to exploit. With only
20 test tasks, absolute rates are imprecise.

The result also explains why adding a cheap model is not automatically useful.
Qwen 8B was inexpensive per token but resolved only 1/32 screen tasks. A cheap
failed run consumes budget without providing a safe substitution or useful
escalation signal.

The cascade refines that lesson. The cheap phase itself cost little and often
reduced strong-model work, but the universal six-call handoff was not
quality-safe. The important unresolved question is whether public task
features or online evidence can predict the 18 safe cases without missing the
two cases that needed fixed Qwen.

## What to run next

No more paid sampling is needed to complete the blog or open-source release,
and repeatedly trying variants on these same outcomes would turn the study
into result chasing.

The next defensible paid study would collect fixed and cascade policies
on-policy on a new training cohort, train a fixed-versus-cascade gate, freeze
it, and evaluate it once on a fresh repository-disjoint cohort. The present 20
labels cannot be used as both training supervision and final evidence. A
diagnostic-only scout with a clean implementation workspace is also a
reasonable preregistered ablation because the tested scout inherited and
modified the same workspace.

## Reproduction

The sanitized result is
[`artifacts/router_baseline_v2_results.json`](../artifacts/router_baseline_v2_results.json).
The exact pre-test classifier is
[`artifacts/router_baseline_v2_task_router.json`](../artifacts/router_baseline_v2_task_router.json).
Raw paid traces and terminal logs remain excluded from the public package.
The sequential protocol and sanitized result are
[`artifacts/sequential_cascade_study_v1.json`](../artifacts/sequential_cascade_study_v1.json)
and
[`artifacts/sequential_cascade_v1_results.json`](../artifacts/sequential_cascade_v1_results.json).

Given compatible local grade artifacts, the key post-collection commands are:

```bash
python scripts/build_router_dataset.py \
  --grades artifacts/pilot_coding_v6_swebench_grades.json \
  --grades outputs/router_baseline_v2/train_calibration_grades.json
python scripts/train_task_router.py
python scripts/freeze_task_router.py
python scripts/evaluate_frozen_task_router.py \
  --grades outputs/router_baseline_v2/test_grades.json
python scripts/analyze_sequential_cascade.py
python scripts/publish_sequential_cascade_results.py \
  --official-report /path/to/official-cascade-report.json
```

Use the published router on new public issue text without a provider call:

```bash
python scripts/route_task.py \
  --repository example/project \
  --issue "Describe the bug to route"
```

The official grader is external to this package. Dataset revision, harness
commit, report checksums, artifact hashes, costs, and final metrics are pinned
in the public result JSON.
