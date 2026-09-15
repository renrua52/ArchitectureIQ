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
4. The answer is checked against GT. If it is wrong, its primary claim is
   rejected immediately: a new claim never enters the KB, while a cited claim
   is removed from the next version.
5. For a correct answer, a new claim goes to the curator, which either maps it
   to an existing ID or returns one canonical proposition. Existing claims gain
   one successful-use count.
6. At epoch end, a new frozen snapshot and its added/reinforced/rejected delta
   are written. New claims are visible to
   the solver only in the next epoch.

The solver and curator configurations are locked by the first epoch, so KB
context is the only model input that evolves. A question ID can be used for
learning only once in a KB, preventing answer leakage across epochs.

`claims.json` contains only active claims. Rejected claims remain auditable in
`events.jsonl` and in earlier immutable snapshots, but never appear in later
solver context.

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

The stable artifacts are:

- `claims.json`: one record per canonical proposition and its evidence counts
- `events.jsonl`: append-only question/claim validation events
- `snapshots/kb_NNNN.json`: the exact KB available to each epoch
- `runs/epoch_NNNN/`: prompts, raw responses, locked solver outputs, and results

Rerunning an interrupted epoch reuses locked solver outputs and completed
question records. Raw solver output is saved before parsing; malformed output
is reformatted without GT and retained alongside the repaired response. Event
IDs make final aggregation idempotent.
