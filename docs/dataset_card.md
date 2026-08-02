# Dataset card: canonical budget-router traces

The planned artifact contains sanitized visible coding-agent traces derived
from SWE-bench Verified. Each turn records task/run/parent/turn identifiers,
policy/model/harness/prompt versions, visible state, eligible actions and
propensities, selected action, remaining budget, conservative and billed
usage, price provenance, latency, errors, tool/test summaries, diff hashes,
and terminal outcome.

No API credentials, environment secrets, raw private keys, hidden model
reasoning, or hidden SWE-bench test evidence may be included. Publication runs
recursive redaction and secret scanning; records are content-hashed. Raw
workspace contents should be published only when upstream repository and
benchmark licenses allow it.

Repository-disjoint train/calibration/test membership is distributed as a
frozen hashed manifest. Checkpoint counterfactuals are training-only and name
their parent run. Censored trajectories include the censoring reason and action
propensities.

The dataset is suitable for policy-specific routing, calibration, cost, and
offline decision research within the observed action support. It should not be
used to claim individual completion guarantees, universal model rankings, or
unbiased counterfactual values outside that support.

