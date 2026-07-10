# GPT-5.6-SOL full-set blind heuristic-formula prompt

You are an independent blind-answer agent for the ArchitectureIQ benchmark.

## Experiment setting

- Model: GPT-5.6-SOL, high reasoning effort.
- You receive the complete 60-question sanitized set at once.
- Answer all 60 questions in one context. You may compare repeated candidates and patterns across the visible questions.
- This is a blind, label-free experiment: no answer or per-question feedback is available while solving.

## Strict isolation protocol

- Use only `artifacts/quiz_attempt_60/questions_sanitized.json` and the instructions in this prompt.
- Do not inspect answer keys, feedback files, scoring files, result summaries, curves, prior attempts, other repository files, or hidden ground-truth artifacts.
- Do not run training, simulations, approximate experiments, data reconstruction, or any command that reveals empirical candidate performance.
- Parameter-count arithmetic and other deterministic calculations from the visible candidate specifications are allowed.
- Do not overwrite or read another agent's output.

## Required heuristic-formula method

Do not jump directly from qualitative intuition to an answer. Build an explicit, inspectable scoring rule for each of these families before finalizing predictions:

1. `bigram_lm`
2. `multivariate_regression`
3. `univariate_regression`

For each family, write a mathematical heuristic of the form

`Score_family(candidate | question) = weighted benefits - weighted risks + interaction terms`.

You choose and explain the features and numeric weights. At minimum consider:

- total trainable parameter count, preferably calculated from the visible architecture rather than guessed;
- the fixed sample budget, batch size, and resulting number of optimizer steps;
- learning rate as a non-monotonic feature: too small can underfit within the budget, while too large can be unstable or overfit, and the useful range may depend on optimizer and model scale;
- optimizer suitability and interactions with learning rate, architecture, normalization, residual connections, and task family; some optimizers may be poor matches even when their nominal learning rate looks attractive;
- capacity versus short-budget trainability, including depth, width, activation, heads, embeddings, residuals, and normalization where relevant;
- loss regularization, weight decay, and the risk of underfitting or overfitting;
- question type: do not reward a feature that is fixed across every option in that question.

The formulas do not need to predict the metric numerically. They must produce comparable scores within each question.

## Required label-free iteration

Use at least two passes without labels or empirical feedback:

1. **Initial pass:** define the three initial formulas, compute every option's feature contributions and total score, and select an initial answer.
2. **Consistency audit:** compare repeated candidate IDs and recurring configurations across all 60 visible questions. Check whether the formulas rank identical candidates consistently, whether a single term dominates implausibly, and whether learning-rate/optimizer/capacity interactions are being double-counted.
3. **Revision pass:** document every formula or weight change. Re-score every question affected by a change. A revision must be based only on visible specifications and cross-question logical consistency, never hidden labels.

You may iterate more than once. Do not silently change weights question by question. Family-level weights must remain shared; explicit question-type interactions are allowed when stated in the formula.

## Complete trace requirement

Write one JSON file to the output path specified in your task message. It must contain:

- `agent`, `model`, `reasoning_effort`, `protocol`, and `source_used`;
- `isolation_confirmation`, including whether any forbidden file was viewed or any empirical experiment was run;
- `parameter_count_formulas`, with the formulas used for MLP and transformer candidates;
- `family_formulas_initial`, including features, numeric weights, interaction terms, and rationale;
- `initial_pass`, with exactly 60 records;
- `consistency_audit`, listing repeated-candidate constraints, anomalies found, and proposed changes;
- `formula_revisions`, listing old value, new value, affected family/questions, and reason for every change;
- `family_formulas_final`;
- `predictions`, with exactly 60 final records;
- `final_reflection`, summarizing which terms drove decisions and the method's main uncertainties.

Each `initial_pass` record must include:

- `n`, `question_id`, and `family`;
- `choice_scores`: one entry for every visible choice with `letter`, `candidate_id`, `num_params`, `steps`, all named feature contributions, `risk_flags`, and `total_score`;
- `initial_letter`, `initial_candidate_id`, `confidence`, and `reason`.

Each final `predictions` record must include:

- `n`, `question_id`, and `family`;
- `initial_letter` and `final_letter`;
- `predicted_candidate_id` matching `final_letter`;
- final score for every choice;
- whether the answer changed during revision and why;
- confidence and a detailed decision trace tied to the final formula.

Use strict JSON with finite numeric values and no Markdown fences. Preserve all intermediate reasoning needed to audit the heuristic; do not replace the trace with a short summary.
