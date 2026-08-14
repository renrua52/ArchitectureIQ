# v1_review — binary classification question review tools

Tools for reviewing v1 LLM binary-classification questions with interactive
curves, model answers, and custom training experiments.

**All runtime outputs go to `{worktree}/data/v1_review/` (gitignored).**
This directory only contains scripts — no data.

## Setup (remote GPU server or local)

```bash
# 1. Clone repo (if not already)
git clone git@github.com:renrua52/ArchitectureIQ.git
cd ArchitectureIQ
git checkout <branch-with-v1_review-tools>

# 2. Create venv + install deps
python3 -m venv .venv
.venv/bin/pip install -e .
# torch with CUDA (adjust for your CUDA version):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Download bundle (v1_llm + datasets)
mkdir -p data/v1bundle
gh release download v1-llm-bundle --repo renrua52/ArchitectureIQ -D /tmp/v1dl
tar xzf /tmp/v1dl/v1_llm.tar.gz -C data/v1bundle
tar xzf /tmp/v1dl/datasets.tar.gz -C data/v1bundle

# 4. Extract binary questions + curves + answers into data/v1_review/
.venv/bin/python tools/v1_review/extract_binary.py
.venv/bin/python tools/v1_review/extract_curves.py
cp tools/v1_review/viewer.html data/v1_review/viewer.html

# 5. Start server (GPU auto-detected)
.venv/bin/python tools/v1_review/train_server.py --port 8502 --bundle data/v1bundle
```

## Access from local machine

```bash
# SSH tunnel (run on local machine)
ssh -L 8502:localhost:8502 ophis-gpu

# Then open in browser:
# http://127.0.0.1:8502/viewer.html
```

## Files

| File | Purpose | Tracked? |
|------|---------|----------|
| `extract_binary.py` | Extract 499 binary questions + LLM answers → `data/v1_review/binary_questions.json` | Yes |
| `extract_curves.py` | Extract per-question learning curves → `data/v1_review/curves/*.json` | Yes |
| `train_server.py` | HTTP server: serves viewer + `/api/train` (GPU training) | Yes |
| `viewer.html` | Interactive viewer (copied to `data/v1_review/` at setup) | Yes |
| `data/v1_review/` | Runtime outputs (binary_questions.json, curves/, answers/) | **No** (gitignored) |
| `data/v1bundle/` | Bundle data (question.json, candidate_spec.json, train.pt/test.pt) | **No** (gitignored) |

## Constraints (AGENTS.md)

- `data/` is gitignored — never commit runtime outputs
- Remote server must not push to git — only clone + run
- Training uses `run_ground_truth` from `architecture_iq.ground_truth.runner` (core invariant: GT from executing generated code)
- Bundle path defaults to `/tmp/v1bundle` or `$V1_BUNDLE` env var
