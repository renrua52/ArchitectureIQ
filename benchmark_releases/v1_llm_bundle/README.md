# V1 LLM + human raw bundle (GitHub Release)

The raw V1 benchmark payloads are **not** stored in git (`benchmarks/`, `data/`
are ignored). This directory only tracks the download manifest pointing at a
GitHub Release.

## Contents (same layout as the maintainer worktree)

| Part | Extracted path | What it is |
|------|----------------|------------|
| `v1_llm` | `benchmarks/v1_llm/` | 1000 questions (`questions/q_*`), `llm_runs/`, reports |
| `v1_human` | `benchmarks/v1_human/` | 250-question human pack (`manifest.json`, `questions.json`) |
| `datasets` | `data/datasets/...` | Full dataset instances referenced by those questions, including **all candidates** and GT (`results/`) |

After download, `benchmarks/v1_human/bake_root/` symlinks are recreated.

## Colleague download

Needs [GitHub CLI](https://cli.github.com/) and access to this repository:

```bash
gh auth login
python tools/benchmark_v1_bundle.py download
```

Optional: `--parts v1_llm datasets` / `--force` to re-fetch / `--skip-extract`.

## Maintainer: pack + upload

```bash
python tools/benchmark_v1_bundle.py pack
python tools/benchmark_v1_bundle.py upload --write-tracked-manifest
git add benchmark_releases/v1_llm_bundle tools/benchmark_v1_bundle.py
git commit -m "Publish V1 LLM bundle GitHub Release manifest."
```

Staging archives live under `artifacts/v1_llm_bundle/` (local only).
Default release tag: `v1-llm-bundle`.
