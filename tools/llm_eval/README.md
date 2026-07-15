# LLM Evaluation Runner

Standalone runner that sends ArchitectureIQ question prompts to an OpenAI-compatible chat API, parses the model's letter answer from an `<answer>` tag, and scores against ground truth.

Does not import `architecture_iq`. Reads question artifacts under `data/` (dataset-scoped runs and legacy `data/questions/`).

## Environment

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
# Optional: use max_completion_tokens instead of max_tokens (never both)
# export OPENAI_MAX_TOKENS_PARAM=max_completion_tokens
```

## Run

```bash
python tools/llm_eval/run.py --model gpt-4o-mini
python tools/llm_eval/run.py data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX --model gpt-4o-mini
```

Optional flags:

- `--temperature 0.0`
- `--max-tokens 16384` (raise if reasoning models stop early)
- `--limit 10` — evaluate first N questions only
- `--workers 4` — concurrent API requests (default 4; use 1 for sequential)
- `--run-dir path/to/run` — explicit output directory
- `--runs-root llm_runs` — parent for auto-named runs
- `--skip-existing` — resume a partial run

## Context-protocol boundary

The runner in this directory is the standalone API evaluation path. The local
`run_sequential_feedback.py` script is retained only to reproduce the historical
2026-07-09 revealed-history comparison; its results are not the canonical score
for the current `main` context workflow.

For current sequential-feedback experiments, use
`tools/sequential_feedback_session.py`. That CLI enforces prediction-before-feedback,
returns the submitted question's correct candidate and per-choice metric values,
and stores an explicit lesson for later questions. This measures online adaptation
under feedback, so report it separately from blind or API-only LLM accuracy.

When comparing any old run, state the exact model, reasoning setting, question
scope, filtering/remapping, and protocol (`blind`, `grouped_feedback`, or
`full_sequential_feedback`). Keep old `llm_runs/` artifacts for provenance, but
do not silently merge them into current `main` results.

## Answer format

Each prompt is augmented with instructions to reason freely, then commit with a tagged answer:

```text
<answer>B</answer>
```

Only responses containing this tag are scored. Prose that mentions "Choice A" or "Choice B" is ignored for parsing.

If the API stops early (`finish_reason: length`), the runner automatically sends up to 3 continuation requests. Result files include `truncated`, `finish_reason`, and `continuation_count` for debugging.

## Output layout

Each run writes to `llm_runs/{timestamp}_{model}/`:

```
run.json                 # model config + accuracy summary
results/
  {question_id}.json      # per-question result (e.g. q_17d258.json)
```

Per-question result fields:

- `ground_truth_letter`
- `parsed_letter`
- `correct`
- `eval_prompt` (full prompt sent to the model, including answer-format instructions)
- `model_response` (full raw LLM text before parsing, including reasoning fields when the API splits them)
- `chain_of_thought` (model response with the `<answer>` tag removed)
- `question_id`, `prompt_hash`, `question_type`, `family`
