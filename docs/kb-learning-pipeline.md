# Prompt-level knowledge accumulation

`tools/kb_pipeline.py` runs an auditable knowledge-learning loop over existing
ArchitectureIQ question artifacts. It does not generate GT or add another
training path: every answer is checked only against the stored
`question.json` produced by the canonical pipeline.

## Data flow

1. An epoch starts from a frozen snapshot in `snapshots/kb_NNNN.json`.
2. PUCT selects one frozen set of at most 20 claims for the whole epoch. Claims
   are shown with only their Laplace-smoothed credibility.
3. The solver returns one answer, one to four weighted evidence claims, and an explanation. The
   response is written to `solver_locked.json` before GT is opened.
4. The answer is checked against GT. Its normalized evidence weights are added
   to `support_count` for a correct answer or `failure_count` for a wrong one.
5. At epoch end, every new claim (including claims from wrong answers) goes to
   the curator in persisted batches of 16. It normalizes each claim and
   deduplicates it against the evolving KB and earlier proposals in the batch.
6. The whole batch is committed once, then a new frozen snapshot and delta are
   written. New claims are visible to the solver only in the next epoch.

The solver and curator configurations are locked by the first epoch, so KB
context is the only model input that evolves. A question ID can be used for
learning only once in a KB, preventing answer leakage across epochs.

`support_count` and `failure_count` are weighted evidence totals. The solver
sees only `(support_count + 1) / (support_count + failure_count + 2)`. PUCT uses
that credibility plus its capped exploration bonus. Existing claims remain in
later snapshots even when their net credit reaches zero or becomes negative.

## Run

The API must implement OpenAI-compatible `POST /chat/completions` and is read
from `VAPI_KEY` and `VAPI_BASE`.

```bash
python tools/kb_pipeline.py init --kb-dir data/kb/my_run
python tools/kb_pipeline.py run-epoch \
  --kb-dir data/kb/my_run \
  --questions-root benchmarks/v1_llm/questions \
  --solver-model claude-opus-5 \
  --curator-model gemini-3.6-flash \
  --workers 6 \
  --skip-seen \
  --limit 50 \
  --injection-limit 20
python tools/kb_pipeline.py show --kb-dir data/kb/my_run
```

To generate fresh v1.5 questions and run successive 50-question epochs until
interrupted:

```bash
python tools/run_kb_learning_forever.py \
  --kb-dir data/kb/my_run \
  --solver-model claude-opus-5 \
  --solver-workers 6
```

To preserve an existing hard-reject run while rebuilding its saved evidence
under the credit rule:

```bash
python tools/kb_pipeline.py migrate-credit \
  --source-dir data/kb/old_run \
  --output-dir data/kb/credit_run
```

This replays the immutable per-question results into new snapshots. It does
not overwrite the source run. Because historical solver prompts came from the
old snapshots, replay corrects the evidence accounting but does not claim to
reconstruct the counterfactual answers a credit-based KB would have produced.

The stable artifacts are:

- `claims.json`: one record per canonical proposition and its evidence counts
- `events.jsonl`: append-only question/claim validation events
- `snapshots/kb_NNNN.json`: the exact KB available to each epoch
- `runs/epoch_NNNN/`: prompts, raw responses, locked solver outputs, and results

Rerunning an interrupted epoch reuses locked solver outputs and completed
question records. Raw solver output is saved before parsing; malformed output
is reformatted without GT and retained alongside the repaired response. Event
IDs make final aggregation idempotent.
