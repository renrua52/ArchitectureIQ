# Tracked question packs

This directory contains immutable, Inspector-ready review pools with their
runtime dependencies and provenance. They are review/blind-practice artifacts,
not canonical blind-evaluation sets.

From the repository root, start the local Inspector:

```powershell
conda run -n architectureiq python -m streamlit run tools/question_inspector/app.py
```

Select a pack in the sidebar or open it directly:

- XOR: `http://localhost:8501/?question_pack=xor-v2.5-100q-37b9da`
- GRU: `http://localhost:8501/?question_pack=gru-v2.5-100q-a48abc`

Each `pack.json` records the collection, data root, profile provenance, and
review semantics. Existing run, question, candidate, and audit artifacts remain
unchanged; `audits/SHA256SUMS` inventories the immutable pack payload.

`data_root` may point at a directory inside the pack (self-contained pack) or,
for packs that reference live generation output, at a clean relative path
inside the repository (e.g. `data` or `benchmarks/v1_llm`). Absolute values
and path traversal are rejected.

The packs contain answer-bearing `question.json` files and are intended for
the current private/internal repository workflow.
