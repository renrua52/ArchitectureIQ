# GPT-5.6-SOL explicit heuristic-formula experiment

This experiment keeps the existing **full-set blind** setting: one model context
sees all 60 sanitized questions, receives no labels or feedback, and cannot run
training or simulations. The only intervention is the prompt: the model must
construct a separate scoring formula for each dataset family, calculate every
choice, audit repeated candidates, revise shared weights without labels, and
record the full structured trace.

## Result

| Condition | Replicates | Scores | Mean | Majority vote |
|---|---:|---|---:|---:|
| Original full-set blind prompt | 6 | 22, 21, 25, 25, 27, 22 / 60 | 39.4% | 24/60 = 40.0% |
| Formula prompt, initial pass | 3 | 20, 22, 19 / 60 | 33.9% | 21/60 = 35.0% |
| Formula prompt, revised final pass | 3 | 18, 21, 20 / 60 | 32.8% | 19/60 = 31.7% |

The formula prompt lowered final mean accuracy by **6.7 percentage points**
relative to the original prompt. Label-free revision also lowered the three-run
total from 61 correct initial predictions to 59 final predictions.

Final family means were 40.0% for `bigram_lm`, 45.0% for
`multivariate_regression`, and 13.3% for `univariate_regression`. The largest
failure was therefore the univariate family, where all three formulas strongly
penalized very large models and conservative learning rates under a 64-step
budget. The benchmark's observed candidate results often reward capacity even
under short budgets, so this plausible qualitative prior became a systematic
error when encoded as a large numeric penalty.

The useful outcome is auditability, not accuracy: every run exposes its parameter
count formulas, numeric feature weights, optimizer/learning-rate interactions,
per-choice scores, risk flags, consistency checks, revisions, and final decision.
The result argues for calibrating weights on a separate training split and then
freezing the formula before evaluating a held-out set. Tuning these weights on
the same 60 labels would contaminate the reported test.

## Files

- `prompt.md` — exact formula-method prompt.
- `agent_A_trace.json`, `agent_B_trace.json`, `agent_C_trace.json` — complete
  60-question initial and final calculation traces.
- `scored_summary.json` — final predictions scored against the hidden key after
  every blind run completed.
- `comparison.json` — baseline, initial, final, family, majority-vote, revision,
  and validation statistics.
- `../blind/prompt.md`, `../blind/summary.json` — original-prompt control.

All three trace files parse as JSON, contain 60 unique initial records and 60
unique final records, contain no missing or invalid answers, and select candidate
IDs consistent with their final letters.
