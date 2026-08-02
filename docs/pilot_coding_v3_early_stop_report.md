# Coding-agent pool v3: early-stop report

## Outcome

The corrected v3 pilot stopped after 20 of 60 planned episodes. Nemotron-3
Nano produced only one structurally valid episode on the first four tasks.
Even if all eight remaining Nano episodes passed, its final score would be
9/12, below the required 10/12.

The other four treatments were structurally valid on all four observed tasks.
No treatment had a provider failure. This distinguishes the v3 result from the
invalid earlier timeout run: the harness and Tinker sampling path worked, while
Nano repeatedly selected an undeclared `str_replace_editor` tool instead of
the harness's declared `bash` tool.

| Model | Episodes | Structurally valid | Provider failures | Best possible final score |
|---|---:|---:|---:|---:|
| Nemotron-3 Nano | 4 | 1 | 0 | 9/12 |
| Qwen3.6 35B-A3B | 4 | 4 | 0 | 12/12 |
| Nemotron-3 Super | 4 | 4 | 0 | 12/12 |
| Inkling | 4 | 4 | 0 | 12/12 |
| Kimi K2.6 | 4 | 4 | 0 | 12/12 |

## Cost

The 20 v3 episodes conservatively cost $9.900319425. Including the previously
reconciled $2.406596660, total conservative exposure is $12.306916085. The
remaining amount under the existing $60 authorization is $47.693083915.

## Decision

Do not run the remaining 40 v3 episodes and do not start the fixed matrix.
Follow the frozen replacement order by testing Qwen 3.5 4B in Nano's price
tier. Preserve the four compatible treatments and resume them only if the
replacement reaches the same 10/12 structural gate.
