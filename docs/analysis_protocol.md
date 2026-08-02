# Frozen analysis protocol

Freeze this file, the repository-disjoint split manifest, model pool, price
snapshot, harness prompt, and environment lock before opening locked test
results. Record their SHA-256 hashes in the run manifest.

The estimand is

\[
q_\pi(x,B)=P_\pi(\text{verified completion within USD budget }B\mid x),
\]

for the named policy, model pool, harness, prompt, and price snapshot. It is not
a task-intrinsic probability or an individual guarantee.

Primary selection excludes every observed hard-cap violator, applies the same
isotonic calibration wrapper to every candidate, requires false-feasible rate
at the 0.8 feasibility threshold to be at most 0.2, and requires integrated
Brier score below the budget-only control. Select maximum normalized
success-versus-budget AUC. If repository-bootstrap intervals overlap, prefer
lower complexity, then lower routing latency and routing cost. If no learned
policy strictly beats the best fixed-model control, keep that control as the
default and report a negative result.

The locked comparison is the selected default against cheapest-only,
strongest-only, best static, and best non-bandit sequential policies at all
five training-derived budgets and three seeds. Report paired
repository-bootstrap intervals. Secondary outcomes are cost per solved task,
mean/p95 cost, latency, tokens, switches, Brier score, log loss, calibration
slope/intercept, reliability, risk–coverage, and false-feasible rate.

Censoring stress tests use both random truncation and truncation dependent on
the terminal outcome. Robustness analyses cover unseen repositories, a 2×
price shock to the training-most-selected model, one model removal, addition
of a held-out model after a fixed probe set, and prompt paraphrases.

