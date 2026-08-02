# Cost-aware task and step routing for budgeted coding agents

## Abstract

We report three sequential, leakage-controlled routing studies. The first
trained a whole-task issue-text router over Qwen, Inkling, and Kimi; it selected
Qwen on every 20-task held-out case and matched the observed oracle at 13/20.
The second tested a universal same-workspace Qwen 8B → Qwen 35B cascade; it
reduced spend by 20.4% but resolved 11/20 instead of 13/20.

The final preregistered follow-up tested isolated GPT-OSS 20B diagnostic
scouting, clean-workspace Qwen finishing, and a learned fixed-versus-isolated
gate. The gate was trained on 35 tasks and frozen before a 39-task
repository-disjoint paired test. Fixed Qwen resolved 15/39 for $19.6834; the
universal isolated policy resolved 13/39 for $25.1480. The gate selected fixed
on all 39 tasks and matched it. Unlike the first study, held-out outcomes
contained real complementarity: four isolated-only and six fixed-only wins.
A non-deployable oracle resolved 19/39 for $19.3033. The central result is
therefore a generalization failure, not an absence of routing opportunity:
training contained zero isolated-only wins, while unseen repositories produced
four.

A fourth, separately frozen experiment moved model selection
inside the agent loop. A binary classifier selected GPT-OSS 20B or Qwen3.6
35B-A3B before every visible model call. On 20 held-out tasks, fixed GPT-OSS
resolved 6 for $3.5905, fixed Qwen resolved 8 for $10.1105, and the router
resolved 9 for $9.5095. The preregistered joint arithmetic criterion passed,
but the router sent all 1,129 calls to Qwen. Its cost difference is therefore
sampling variation, not a routing saving.

A fifth prospectively frozen follow-up calibrated the unchanged classifier
family on those already-visible prefixes without using their terminal
outcomes, added activation guards, passed a 20-task development gate, and
tested three policies on 60 new tasks. Fixed GPT-OSS resolved 13/60 for
$9.5290, fixed Qwen 24/60 for $30.9780, and the guarded router 26/60 for
$24.5376. It used GPT-OSS on 12.70% of calls across 56/60 trajectories. On
the primary 50-task novel-repository subgroup, it tied Qwen at 22/50 while
saving 20.80%. The full paired quality interval was [-6.67, +13.33] points,
so quality superiority is not established; the cost-saving interval was
[9.91%, 30.57%].

## First agent-step experiment

Amendment 007 froze the two-tier model pool, public TwinRouterBench revision,
instance-grouped train/calibration/static-test split, hashed-logistic
classifier, Platt calibration, threshold rule, static launch gate, 20-task
SWE-bench cohort, three treatments, official grader, and budget before new
paid inference.

The frozen artifact passed its sealed static gate: 97.30% strong-tier recall,
15.82% cheap call share, and 98.77% trajectory pass. All 60 live held-out
episodes then ran before official grades were opened.

| Policy | Submitted | Resolved | Total cost |
|---|---:|---:|---:|
| Fixed GPT-OSS 20B | 10/20 | 6/20 | $3.5905 |
| Fixed Qwen3.6 35B | 11/20 | 8/20 | $10.1105 |
| Frozen per-call router | 13/20 | 9/20 | $9.5095 |

Router versus fixed Qwen had three router-only and two fixed-only
resolutions, exact McNemar \(p=1.0\), and a paired-bootstrap quality interval
of [-15, +25] percentage points. The observed 5.94% cost difference had a
bootstrap interval of [0.0008%, 12.21%].

Mechanistically, however, the router made 0 cheap calls, 1,129 strong calls,
and 0 switches. Both policy arms therefore used Qwen throughout, with
independent trajectory seeds. The prespecified endpoint is reported as met,
while the intended mixed-model mechanism is reported as not activated.

Two official errors were confirmed as malformed submitted model output, one
under fixed Qwen and one under the router. They are classified, preserved, and
counted unresolved. No provider failures or unclassified grader errors
remain.

The development smoke and held-out treatments cost $23.461945320 against a
frozen $43.90 maximum. Full design, interpretation, and hashes are in
`docs/agent_step_router_v1_final_report.md` and
`artifacts/agent_step_router_v1_results.json`.

## Guarded agent-step follow-up

Amendment 008 treated the earlier 20-task result as development evidence and
froze a new protocol before new paid inference. The classifier was refit on
459 public training rows, calibrated on 112 rows, and assigned a live
threshold using the prior visible prefixes only. Initial-strong,
maximum-cheap-burst, context, and strong-dwell guards bounded the policy.

The unchanged artifact then resolved 11/20 development tasks while using
GPT-OSS on 11.19% of calls and passed every activation, quality, structural,
and provider gate. Sixty new tasks were frozen: 50 from Astropy, Django, and
Matplotlib repository families excluded from fitting and threshold
development, and 10 from familiar repositories. All 180 episodes completed
before official labels were opened.

| Policy | Submitted | Resolved | Total cost |
|---|---:|---:|---:|
| Fixed GPT-OSS 20B | 40/60 | 13/60 | $9.5290 |
| Fixed Qwen3.6 35B | 42/60 | 24/60 | $30.9780 |
| Guarded per-call router | 44/60 | 26/60 | $24.5376 |

Router versus Qwen had five router-only and three Qwen-only resolutions,
exact McNemar \(p=0.7266\), and a paired quality interval of
[-6.67, +13.33] points. The router saved $6.4404, or 20.79%, with a paired
cost-saving interval of [9.91%, 30.57%].

The intended mechanism activated: 364 GPT-OSS calls and 2,502 Qwen calls,
12.70% cheap share, cheap use on 56/60 trajectories, 400 switches, and no
guard violations. Repricing the exact cheap-call tokens at Qwen rates gives a
$1.9501 direct list-price saving; the full policy-level saving also includes
changed trajectory length and behavior.

On the 50-task primary transfer subgroup, both router and Qwen resolved 22
tasks. The router saved 20.80%; paired intervals were [-10, +10] quality
points and [11.34%, 30.13%] cost saving. Equal repository weighting was less
favorable on quality because the router led on Django but trailed on the
smaller Astropy and Matplotlib aggregate; only three novel repository
clusters make that estimate imprecise.

One fixed-Qwen diff was confirmed malformed and counted unresolved. Two
router image-pull timeouts were retried resume-safely through the pinned
harness; one resolved and one completed unresolved. No final infrastructure
or unclassified grader errors remain.

The result passes the predeclared joint endpoint and supports observed
fixed-Qwen-compatible quality at lower cost. It does not prove superiority or
a narrow non-inferiority margin. Full details and hashes are in
`docs/agent_step_router_v2_final_report.md` and
`artifacts/agent_step_router_v2_results.json`.

## Estimand

For public task text \(x\), hard per-episode budget \(B\), frozen treatment
pool \(M\), and static selection policy \(\pi\), the primary estimand is

\[
P(Y_{\pi(x)}=1,\ C_{\pi(x)}\le B),
\]

where \(Y\) is official terminal resolution and \(C\) is conservative uncached
list-price spend. A no-patch episode is unresolved. Results are policy- and
treatment-specific, not individual guarantees or universal model rankings.

## Frozen design

The study manifest is
`artifacts/router_baseline_study_v2.json`, hash
`caec3eab9575b53f9fef2bbb670e8292186b71857798d37575d38c2b20ec156d`.
It pins:

- SWE-bench Verified revision
  `91aa3ed51b709be6457e12d00300a6a596d4c6a3`;
- a seeded 60/20/20 train/calibration/test allocation;
- repositories disjoint across the three splits;
- treatment renderer, reasoning, context, and sampling settings;
- $0.35 hard episodes for Qwen 8B and $0.90 for other treatments;
- screen continuation and pruning rules;
- a $190 incremental working limit and $200 absolute limit.

The official terminal harness was pinned at commit
`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`. The exact local dataset
snapshot had SHA-256
`545d417f42f5bff2112f8a4d7b283d51fd23c17d0805fc910046732c167a045c`.

## Screen

The initial treatments were Qwen/Qwen3-8B, Qwen/Qwen3.6-35B-A3B,
thinkingmachines/Inkling, and moonshotai/Kimi-K2.6. The 32-task screen combined
12 frozen pilot tasks with 20 newly sampled training tasks.

Qwen 35B resolved 13/32 tasks. The task oracle resolved 15/32, producing the
two-task, 6.25-point gain required by the complementarity gate. Qwen 8B
produced structurally valid traces on 31/32 tasks but resolved only 1/32, giving
one safe cheap substitution against a threshold of three. It was pruned.
Qwen 35B, Inkling, and Kimi survived according to the frozen retain rules.

The screen gate hash is
`22ae350bf29df31eb98840ad732e7f7f6f22a43347e06b281bf5023b0ffe174b`.

## Supervision and model

The final paired supervision matrix contained 240 rows: three model outcomes
for each of 60 training and 20 calibration tasks. The 60 training tasks
combined 12 reused pilot tasks, 20 screen tasks, and 28 expansion tasks.
Official terminal status supplied binary labels.

The router fits an independent balanced logistic head per model. It maps up to
4,000 issue tokens to a 1,024-dimensional signed feature hash containing
unigrams, adjacent bigrams, and a coarse text-length bucket. Features are L2
normalized. Repository identity, gold patch, FAIL_TO_PASS, PASS_TO_PASS,
grader logs, and test outcomes are excluded.

Expected per-model cost is the training mean. The selector chooses

\[
\arg\max_m \left[
  \hat p_m(Y=1\mid x) -
  \lambda\frac{\mathbb E[C_m]}{\max_j\mathbb E[C_j]}
\right].
\]

The search covered learning rates 0.2 and 0.5, L2 values 0.001, 0.01, and 0.05,
and cost penalties 0, 0.05, 0.1, 0.2, 0.4, and 0.8. Selection maximized
calibration resolved count, then minimized observed selected cost, then Brier
score, with simpler cost settings as later tie-breaks.

The chosen fit used learning rate 0.5, L2 0.001, 200 epochs, and cost penalty
0.8. It selected Qwen for all 20 calibration tasks, resolving 11 at a mean
cost of $0.4244 with a multi-head Brier score of 0.2338.

The artifact hash is
`e60eeefa2f7428880603421d2fcf68013eb45181159dafb5ad9d907c1bfcb0f6`;
its file SHA-256 is
`4cfbe081bc2cce37c81bc5ecbb49e65279a60f9edcc0e1df42838ac3166fb896`.
The pre-test freeze hash is
`34ef59db74012923111fe835e76b2d5c0d41cfeefd14769f90249fa780a79a25`.

## Held-out test

Each surviving model independently attempted the same 20 held-out tasks. Of 60
episodes, 35 submitted patches and 25 did not. All 35 submissions received a
final official grade with zero infrastructure errors.

| Fixed treatment | Submitted | Resolved | Rate | 95% Wilson interval | Mean cost | Cost / resolved |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B-A3B | 16 | 13 | 0.65 | [0.433, 0.819] | $0.4219 | $0.6490 |
| Inkling | 13 | 10 | 0.50 | [0.299, 0.701] | $0.5197 | $1.0394 |
| Kimi K2.6 | 6 | 4 | 0.20 | [0.081, 0.416] | $0.6486 | $3.2430 |

All Inkling and Kimi successes were Qwen successes. Paired exact McNemar
comparisons were:

- Qwen versus Inkling: three Qwen-only, zero Inkling-only, \(p=0.25\);
- Qwen versus Kimi: nine Qwen-only, zero Kimi-only, \(p=0.00390625\);
- Inkling versus Kimi: six Inkling-only, zero Kimi-only, \(p=0.03125\).

These p-values are descriptive and unadjusted. The small sample is better
represented by the wide effect intervals than by binary significance labels.

The frozen router selected Qwen on 20/20 tasks. It resolved 13, cost $8.4376 in
total, and cost $0.6490 per resolution. Its paired quality and cost deltas
against the best fixed policy are exactly zero. The hindsight oracle also
resolved 13.

## Secondary sequential cascade

The source manifest preregistered an optional on-policy cascade arm and an $18
maximum reservation before static test collection. After the static result,
but before any cascade outcome, we froze the exact arm in
`artifacts/sequential_cascade_study_v1.json`, manifest hash
`3ebd1f7c6d893a2d14174deeb8273a519ede433cb79008bc2287cb72abdfa8f3`.
This timing makes the arm a secondary exploratory paired experiment, not a
second confirmatory primary endpoint.

The frozen treatment used:

- Qwen 3 8B as a six-call, $0.06-capped scout;
- Qwen 3.6 35B-A3B as the finisher;
- one shared mutable workspace and the complete visible transcript;
- no transfer of hidden reasoning;
- the same 75-step and $0.90 total limits as fixed Qwen;
- the existing fixed-Qwen episode and official grade as paired comparator.

All 20 episodes executed the handoff. The scout never submitted early. It made
120 calls for $0.149081595; the finisher made 711 calls for $6.567628200.
Seventeen episodes submitted, and the official harness graded all 17 with zero
errors.

| Policy | Submitted | Resolved | Rate | 95% Wilson interval | Total cost | Cost / resolved |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Qwen 35B | 16 | 13 | 0.65 | [0.433, 0.819] | $8.4376 | $0.6490 |
| Qwen 8B → Qwen 35B | 17 | 11 | 0.55 | [0.342, 0.742] | $6.7167 | $0.6106 |

The paired resolution delta was -0.10 with a 10,000-sample task-bootstrap 95%
interval of [-0.25, 0.00]. There were two fixed-only resolutions and zero
cascade-only resolutions; the two-sided exact McNemar \(p\)-value was 0.5.
The total cost delta was -$1.720925775 with a bootstrap interval of
[-$3.8772, $0.4142]. The cascade therefore failed the frozen rule requiring
more resolutions, or equal resolutions at lower total cost.

The cascade was cheaper on 12 tasks and matched the fixed outcome on 18. A
hindsight quality-preserving policy chooses the successful treatment when
outcomes disagree and the cheaper treatment otherwise. It selects cascade on
11 tasks and fixed on 9, resolves 13, and costs $5.881002060—a 30.3% reduction
from fixed. This is an oracle upper bound that uses terminal outcomes, not a
trained or deployable stage gate.

## Isolated diagnostic-scout policy gate

We next froze an isolated-stage study before any new provider call. GPT-OSS
20B inspected a disposable workspace for six calls under a $0.08 sub-cap.
Only sanitized visible evidence was transferred to a Qwen 35B finisher in a
fresh workspace under the remainder of a shared $0.90 cap. Hidden reasoning,
scout patches, and scout workspace mutations were not transferred.

Candidate screening and expansion created a 35-task training cohort across
five repositories. The isolated policy resolved 11 tasks versus fixed Qwen's
16, produced zero unique wins versus five fixed-only wins, and cost $21.5427
versus $17.9206. A balanced hashed-logistic gate, validated
leave-one-repository-out, consequently selected fixed on every fold. Its
artifact and SHA-256 were frozen before test collection.

The held-out cohort used 39 tasks from six disjoint repositories:

| Policy | Submitted | Resolved | Total cost | Cost / resolved |
|---|---:|---:|---:|---:|
| Fixed Qwen 35B | 28 | 15/39 | $19.6834 | $1.3122 |
| Isolated GPT-OSS 20B → Qwen 35B | 19 | 13/39 | $25.1480 | $1.9345 |
| Frozen gate | 28 | 15/39 | $19.6834 | $1.3122 |
| Hindsight oracle | — | 19/39 | $19.3033 | $1.0160 |

The isolated-versus-fixed paired delta was -5.13 percentage points with a
bootstrap 95% interval of [-20.51, +10.26] points. Four tasks were
isolated-only and six fixed-only; exact McNemar \(p=0.7539\). The oracle's
four incremental resolutions demonstrate policy complementarity, while the
gate's all-fixed behavior demonstrates that issue-text supervision did not
generalize across the repository shift.

The scout itself cost only $0.2264, 0.90% of isolated-policy spend. Fresh Qwen
finishers cost $24.9216. The complete isolated policy was cheaper on 13 tasks
but more expensive on 26, so cheap scouting did not automatically lower total
cost.

## Cost accounting

The follow-up produced 296 new valid model episodes for recorded conservative
exposure of $158.708486065. One provider call hung and was abandoned before a
valid row was recorded; reserving its full $0.90 cap gives incremental
reserve-adjusted static exposure of $159.608486065. The sequential arm added
$6.716709795, bringing current reserve-adjusted exposure to $166.325195860.

The router training matrix reused a prior 36-episode, three-model pilot costing
$38.793438190. Valid observations used by the static and sequential experiments
therefore represent $204.218634050 of conservative execution, of which the
reused portion was not charged during this follow-up.

The isolated-stage study is accounted separately. It cost $10.072174440 for
screening, $18.238539555 for expansion, and $44.831404290 for the paired test:
$73.142118285 total against a frozen $112.50 ceiling.

## Validity

Strengths include paired fixed-model outcomes, hard-cap accounting, explicit
no-patch failures, a separate calibration split, a cryptographic pre-test
freeze, repository-disjoint held-out evaluation, a pinned public dataset and
official harness, and a task oracle that exposes the maximum attainable static
routing quality on observed outcomes.

Threats include one stochastic seed per treatment, only 20 test tasks, custom
budgets and treatment settings, provider and price drift, and selection of the
study after an earlier compatibility pilot. The screen's two-task training
oracle gain did not reproduce on held-out repositories. Exact classifier
probabilities should not be interpreted as individually calibrated guarantees.

The sequential arm has additional limitations. Its phase policy is a fixed
six-call heuristic, not a learned router; scout and finisher shared a mutable
workspace; and the same 20 tasks were reused for the paired secondary arm.
The post-hoc fixed/cascade oracle is useful only as evidence that an observed
cost-quality boundary existed. Training on those outcomes and evaluating on
the same tasks would be leakage.

The isolated-stage arm removes workspace sharing and uses a fresh
repository-disjoint 39-task test, but it still has one stochastic seed and
wide uncertainty. Its gate was trained on only 35 tasks, with no positive
isolated-only example. The four such outcomes in unseen repositories should
be interpreted as evidence of distribution shift and a target for new
training—not as permission to refit and retest on the same labels.

Two training/calibration submissions were terminal malformed-patch failures in
the harness and are counted unresolved; they were not infrastructure failures.
The final test reports contain zero errors.

This is not an official public SWE-bench leaderboard score.

## Conclusion

The whole-task router and same-workspace cascade did not beat fixed Qwen. The
isolated-stage follow-up went further: it trained and held-out tested a real
policy gate. That gate safely matched fixed Qwen, while the held-out oracle
revealed four additional attainable resolutions.

This is the strongest routing evidence in the project, but not a deployable
routing win. The observed complementarity appeared only in repositories absent
from training. A defensible next study should train a dynamic gate on
scout-visible runtime evidence and evaluate it once on new repository-disjoint
tasks. The present 39 outcomes cannot serve as both training labels and final
evidence.

## Artifacts

The sanitized public result is
`artifacts/router_baseline_v2_results.json`. The exact pre-test classifier is
`artifacts/router_baseline_v2_task_router.json`. The former includes study and
router hashes, official report checksums, test metrics, intervals, and cost
accounting; the latter contains only the frozen hashed-linear heads and
selector. Raw paid trajectories, reasoning, patches, and terminal logs are
excluded pending independent license, privacy, and secret review.

The sequential freeze is `artifacts/sequential_cascade_study_v1.json`, and its
sanitized result is `artifacts/sequential_cascade_v1_results.json`. The result
pins official report, grade, evaluation, and protocol hashes plus phase costs,
paired statistics, task-level terminal outcomes, and the non-deployable policy
oracle. It excludes raw trajectories, patches, reasoning, terminal logs, and
credentials.

The isolated-stage protocol is
`artifacts/isolated_stage_router_study_v1.json`, its exact pre-test gate is
`artifacts/isolated_stage_v1_task_gate.json`, and its sanitized result is
`artifacts/isolated_stage_v1_results.json`. The detailed final report is
`docs/isolated_stage_v1_final_report.md`.
