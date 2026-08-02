# Qwen 3.5 4B replacement pilot: early-stop report

## Outcome

Qwen 3.5 4B did not pass the structural gate. The pilot stopped after seven
of 12 tasks with four valid episodes and three failures. The best possible
final result was then 9/12, below the required 10/12.

All three failures had the same cause. The model used the declared shell tool
and continued across many turns, but eventually accumulated a 65,565-token
visible prompt. Adding the frozen 8,000-token output allowance exceeded the
treatment's 65,536-token context window, so Tinker rejected the next request.
The failures are counted because context length is part of the pinned
model-and-harness treatment.

| Task | Structural result | Terminal reason | Conservative cost |
|---|---:|---|---:|
| Matplotlib 24570 | Fail | Context-window rejection | $0.254772330 |
| Astropy 14365 | Pass | Submitted | $0.115152360 |
| Seaborn 3187 | Fail | Context-window rejection | $0.646606665 |
| Django 14765 | Pass | Submitted | $0.207104295 |
| Pylint 4661 | Pass | Submitted | $0.112691835 |
| Matplotlib 24637 | Pass | 75-turn limit | $0.456086055 |
| Astropy 12907 | Fail | Context-window rejection | $0.665653320 |

## Cost and decision

The seven episodes cost $2.458066860. Total conservative project exposure is
$14.764982945, leaving $45.235017055 under the existing $60 authorization.

Stop this candidate and test the second frozen replacement,
`openai/gpt-oss-120b:peft:131072`. Its 131,072-token context directly tests
whether the observed incompatibility is specific to the smaller context
treatment.
