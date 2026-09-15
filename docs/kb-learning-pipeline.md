# Prompt-level knowledge accumulation

`tools/kb_pipeline.py` runs an auditable knowledge-learning loop over existing
ArchitectureIQ question artifacts. It does not generate GT or add another
training path: every answer is checked only against the stored
`question.json` produced by the canonical pipeline.

## Data flow

1. An epoch starts from a frozen snapshot in `snapshots/kb_NNNN.json`.
2. A small, lexically relevant claim slice is added to each benchmark prompt.
3. The solver returns one answer, one primary claim, and an explanation. The
   response is written to `solver_locked.json` before GT is opened.
4. The answer is checked against GT. A cited existing claim gains one credit
   for a correct answer and loses one credit for a wrong answer. Claims are not
   deleted. A wrong new claim is recorded but never enters the KB.
5. For a correct answer, a new claim goes to the curator, which either maps it
   to an existing ID or returns one canonical proposition. Existing claims gain
   one successful-use count and one credit.
6. At epoch end, a new frozen snapshot and its added/reinforced/rejected delta
   are written. New claims are visible to
   the solver only in the next epoch.

The solver and curator configurations are locked by the first epoch, so KB
context is the only model input that evolves. A question ID can be used for
learning only once in a KB, preventing answer leakage across epochs.

Each stored claim has `support_count`, `failure_count`, and
`credit = support_count - failure_count`. Existing claims remain in later
snapshots even when their credit reaches zero or becomes negative. Wrong new
claims remain auditable in `events.jsonl` but do not enter the KB.

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
  --workers 5 \
  --skip-seen \
  --limit 20
python tools/kb_pipeline.py show --kb-dir data/kb/my_run
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
