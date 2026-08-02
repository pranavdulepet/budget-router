# Amendment 007: confirmatory agent-step routing

Status: **frozen before classifier fitting or new paid inference**

Effective date: 2026-07-29

Parent study: `open-router-study-2026-07-29-v1`

## Authorization and inferential status

The user explicitly restored agent-level routing as the primary project goal
after reviewing the completed one-shot public study. This amendment opens a
new confirmatory agent-step experiment. It does not alter, replace, or
reinterpret the earlier one-shot confirmatory result.

No held-out outcome from the 20-task active-router manifest has been opened.
Those tasks were frozen before the current amendment and no model has executed
them. The Tinker authorization remains a hard incremental ceiling of $220; it
is not a spending target.

## Research question and estimand

At every LLM call in a coding-agent trajectory, can a frozen classifier choose
between a cheap model and a stronger model so that the resulting policy:

1. resolves at least as many held-out SWE-bench Verified tasks as the fixed
   stronger model; and
2. has lower total conservative observed inference cost?

The unit of the primary quality analysis is the SWE-bench task. The routing
decision is conditional on the router-visible trajectory prefix immediately
before an LLM call. The router selects the model only; the mini-swe-agent
scaffold and tool policy are shared by all treatments.

## Frozen model pool

- Cheap tier: `openai/gpt-oss-20b`
- Strong tier: `Qwen/Qwen3.6-35B-A3B`

Both models passed the project's official tool-use eligibility screen. Models
that failed structural or official-resolution gates remain excluded. The
binary pool intentionally spans two model families and a threefold published
input/output price ratio. It is a two-tier agent-routing experiment, not a
claim about optimal routing over every Tinker model.

Renderer, sampling, context, and price settings are inherited unchanged from
`configs/model_pool_router_v1.json` and
`configs/tinker_prices_2026-07-29_router_v1.json`.

## Public training source

Use the Apache-2.0 TwinRouterBench static question bank:

- repository: `https://github.com/CommonstackAI/TwinRouterBench`
- commit: `430acecac71141de77afd8e5e13690d236d58e93`
- question-bank SHA-256:
  `5b4f90c24643b214a9b0f26bf4e05afc742554262f4ef405e0b3b4a4cce503f4`
- manifest SHA-256:
  `e575b8cc8e33bba993f2d1bcf09b4ee6940fbb098c9255a9c8e5ef7c6771e726`
- released rows: 970 router-visible prefixes from 520 instances

Map `target_tier_id == 0` to the cheap tier and every higher target tier to the
strong tier. This mapping asks whether the current call needs more than the
released low tier; it does not assert that the two Tinker models are identical
to TwinRouterBench's vendor pool.

For leakage resistance, exclude all TwinRouterBench SWE-bench rows whose
instance repository prefix appears in the dynamic held-out repositories:
`pydata`, `pytest-dev`, `scikit-learn`, `sphinx-doc`, or `sympy`.

Split the remaining rows by `instance_id`, never by row, within benchmark.
Order instances by SHA-256 of
`agent-step-router-20260729-v1:<benchmark>:<instance_id>` and assign 70% to
training, 15% to calibration, and the remainder to static test, with integer
boundaries computed per benchmark. No trajectory may cross splits.

## Frozen classifier and routing features

The deployable classifier is binary hashed logistic regression with:

- deterministic 16,384-dimensional signed unigram/bigram hashing;
- hash seed `agent-step-router-features-v1`;
- class weighting from the training split;
- scikit-learn logistic regression with the deterministic `liblinear` solver,
  balanced class weights, and at most 2,000 solver iterations;
- inverse L2 strength `C` selected from `{0.1, 1.0, 10.0}` by five-fold
  instance-grouped training-only cross-validation;
- selection metric: mean binary log loss, then lower `C` as the deterministic
  tie-break;
- Platt calibration fit only on the calibration split.

The text rendered for each decision contains only router-visible information:
the original user request, the most recent visible messages/tool results, the
current step and message counts, tool-call and tool-result counts, recent
return-code/error indicators, approximate context length, and coarse
code/question indicators. It excludes hidden reasoning, reference patches,
official tests, held-out labels, and future messages.

Choose the largest calibrated probability threshold that maximizes cheap
routes on calibration while preserving both 100% strong-row recall and 100%
trajectory pass under the static tier predicate. Ties choose the lower
threshold. The classifier, calibrator, threshold, source hashes, and split IDs
must be serialized and hashed before static-test evaluation.

## Static launch gate

Open the static test only after freezing the artifact. Dynamic collection may
begin only if:

- static-test trajectory pass is at least 90%;
- static-test strong-row recall is at least 95%;
- the cheap-route share is at least 10%; and
- the artifact and source hash checks pass.

Failure stops paid dynamic collection. Static-test results may be reported but
may not be used to change the frozen classifier for this confirmatory run.

## Runtime policy

Before every LLM call, the classifier scores the current visible prefix.

- Select GPT-OSS 20B when calibrated `P(needs_strong)` is at or below the
  frozen threshold.
- Select Qwen 35B otherwise.
- Force Qwen when the approximate prompt length exceeds 24,000 tokens, the
  remaining budget cannot reserve the cheap call safely, or input validation
  fails.
- Escalation to Qwen is immediate. After a Qwen call, require two complete
  Qwen calls before de-escalating, preventing one-step oscillation.
- A cheap-provider failure may fail over once to Qwen on the unchanged visible
  prefix; its conservative reservation remains charged.
- Every decision records score, calibrated probability, reason, selected
  model, cumulative cost, and visible feature summary.

The per-episode hard cap is $0.90 for the routed and fixed-strong policies and
$0.35 for fixed cheap. Every call is preflight-reserved using uncached public
list prices.

## Dynamic held-out design

Use the 20 untouched tasks in
`artifacts/active_router_task_manifest.json`. They span five repositories that
are disjoint from the original active-router development repositories.

Run a complete paired block for every task:

1. fixed GPT-OSS 20B;
2. fixed Qwen 35B; and
3. the frozen per-step classifier router.

Treatment order within each task is deterministically randomized from seed
`20260729`. Use the same task ordinal seed schedule, prompt, scaffold, maximum
75 agent steps, Docker image, and official SWE-bench dataset revision across
treatments. Do not inspect official task grades until all 60 canonical
episodes are complete. Zero-call infrastructure failures may be retried under
the existing recovery rule; quality failures may not.

One routed-policy development smoke episode is permitted before held-out
collection, on an already-used development task. It is for structural and
budget validation only and cannot contribute a quality label or classifier
feature.

## Analysis

Primary joint success requires:

- routed resolved count greater than or equal to fixed Qwen resolved count;
  and
- routed total conservative observed cost lower than fixed Qwen total cost.

Report all fixed and routed counts and costs regardless of success. Also
report paired task bootstrap 95% intervals with 10,000 resamples and seed
`20260729`, McNemar discordant counts, cost per resolved task, route share,
switch count, forced-strong reasons, model calls by trajectory position, and
provider/infrastructure failures. A hindsight per-step or per-task oracle is
non-deployable and descriptive only.

The public TwinRouterBench dynamic results may be cited as an external
reference, not pooled with or treated as a same-split baseline.

## Budget and stop rules

Recorded prior active-router exposure is $43.277878155, including the full
$4.80 conservative reserve for interrupted requests.

- one development smoke: at most $0.90;
- 20 fixed-cheap episodes: at most $7.00;
- 20 fixed-strong episodes: at most $18.00;
- 20 routed episodes: at most $18.00;
- maximum new exposure under this amendment: $43.90;
- maximum cumulative active-router exposure after this amendment: 
  $87.177878155.

Stop before a call that could exceed either its episode cap, the amendment
cap, or the original $220 ceiling. Stop and investigate after two consecutive
provider failures, any artifact/source hash mismatch, held-out grade leakage,
or an inability to reproduce the frozen static artifact.
