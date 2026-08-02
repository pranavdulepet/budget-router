# Final coding-agent compatibility pilot

## Outcome

The frozen v6 final pool completed all 60 planned compatibility episodes:

- 12 training-only SWE-bench Verified tasks.
- Five model treatments.
- A conservative USD hard cap of $0.90 per episode.
- A maximum of 8,000 output tokens per turn.
- A 300-second timeout per model request.
- A 75-turn episode limit.

All five treatments produced 12 of 12 structurally valid episodes. The
compatibility gate therefore passed.

Structural validity means that the treatment:

1. Emitted a declared `bash` tool call.
2. Supplied valid tool arguments.
3. Received a tool result.
4. Continued after receiving a tool result.
5. Produced a valid terminal trace.

It does not mean that the episode submitted a patch or solved the SWE-bench
task.

## Final-pool results

| Treatment | Structurally valid | Submitted patches | Mean conservative cost | Total conservative cost | Mean latency | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Inkling | 12/12 | 5/12 | $0.531822463 | $6.381869560 | 75.2 s | 47.88 tasks/hour |
| Qwen 3.6 35B-A3B | 12/12 | 8/12 | $0.516424681 | $6.197096175 | 275.0 s | 13.09 tasks/hour |
| Kimi K2.6 | 12/12 | 3/12 | $0.707305286 | $8.487663435 | 341.4 s | 10.55 tasks/hour |
| GPT-OSS 120B 128K | 12/12 | 0/12 | $0.787818125 | $9.453817500 | 634.1 s | 5.68 tasks/hour |
| Nemotron-3 Super | 12/12 | 3/12 | $0.689415960 | $8.272991520 | 1,488.8 s | 2.42 tasks/hour |

There were no provider failures in the 60 final-pool episodes.

Terminal states explain why structural validity is not task success:

| Treatment | Submitted | Budget limit | Other agent limit |
|---|---:|---:|---:|
| Qwen 3.6 35B-A3B | 8 | 2 | 2 |
| Inkling | 5 | 7 | 0 |
| Kimi K2.6 | 3 | 9 | 0 |
| Nemotron-3 Super | 3 | 2 | 7 |
| GPT-OSS 120B 128K | 0 | 10 | 2 |

Budget- and agent-limit exits still count as structurally valid because the
agent used the tool protocol correctly and produced a well-formed terminal
trace. They count as unresolved for coding-quality evaluation.

## Cost accounting

- Final-pool 60-episode pilot: **$38.793438190**
- Earlier diagnostics, invalid seed-timeout run, eliminated candidates, smoke
  checks, and the conservatively reserved censored episode:
  **$6.296908890**
- Total conservative project exposure: **$45.090347080**
- Approved pilot budget: **$60.00**
- Unused pilot authorization: **$14.909652920**

The accounting uses conservative uncached list-price estimates. It is the
budget-safety ledger, not a claim about the provider's final invoice.

## Official terminal grades

The 19 submitted patches were exported from their saved trajectories and
evaluated with the official SWE-bench Docker harness. Episodes without a patch
count as unresolved.

- Dataset: `SWE-bench/SWE-bench_Verified`
- Dataset revision: `91aa3ed51b709be6457e12d00300a6a596d4c6a3`
- SWE-bench harness commit: `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`
- Submitted patches completed by the grader: **19/19**
- Grader errors: **0**

| Treatment | Submitted | Resolved | Resolution / 12 | Pass / submitted | Cost / resolved | Episode latency / resolved |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B-A3B | 8 | 6 | 50.0% | 75.0% | $1.0328 | 550.1 s |
| Inkling | 5 | 4 | 33.3% | 80.0% | $1.5955 | 225.6 s |
| Kimi K2.6 | 3 | 3 | 25.0% | 100.0% | $2.8292 | 1,365.5 s |
| Nemotron-3 Super | 3 | 1 | 8.3% | 33.3% | $8.2730 | 17,865.3 s |
| GPT-OSS 120B 128K | 0 | 0 | 0.0% | — | — | — |
| **Total** | **19** | **14** | **23.3%** | **73.7%** | **$2.7710** | **2,412.4 s** |

“Episode latency / resolved” divides all 12 episode-seconds for a treatment by
its resolved count. It therefore includes time spent on failures and
non-submissions. This is the appropriate opportunity-cost measure for a fixed
treatment; it is not the mean runtime of successful episodes.

The 14 model-task resolutions cover six unique tasks. Qwen resolved all six,
so an oracle over the observed treatments adds zero unique resolutions over
the best fixed treatment in this pilot.

The previously designed 500 × 5 × 3 matrix has not started and is not a next
step for this project. Its protocol remains in the repository as a
preregistered design artifact only.

## Interpretation

The pilot establishes that the final five treatments can repeatedly operate
the frozen Tinker + renderer + mini-swe-agent tool protocol without provider
failure. The official grades also establish task outcomes for the 60 pilot
episodes.

The small training-only sample does not support a universal model ranking or a
learned-router claim. It does show that protocol success substantially
overstated task success, that non-submission was the dominant failure mode, and
that the observed pool supplied no unique quality win beyond Qwen on these 12
tasks. GPT-OSS produced structurally valid interaction traces in all 12 tasks
but no submitted patch. Nemotron-3 Super was structurally valid but much
slower than every other treatment.

## Recommended next step

Publish the router as an infrastructure and replay reference with this pilot
clearly labeled exploratory. Use public or contributor-supplied logged
outcomes for no-provider-cost baselines, keep Qwen as the pilot fixed-model
control, and require a new repository-disjoint evaluation before claiming that
any learned router improves on it. Do not start the fixed matrix.
