# ArchitectureIQ

A prototype benchmark for the **modeling intuition** of LLMs (and humans): given a dataset instance and several **candidates** (model + optimizer + loss + budget), pick which **choice** achieves the best selection metric after its stated training budget.

Design: [plan-v2.md](./docs/architecture/plan-v2.md) · Terminology: [AGENTS.md](./AGENTS.md#terminology) · Product development: [docs/product-development.md](./docs/product-development.md)

Interactive experiment report: [README.html](./README.html) (Chinese).

## AI agent instructions

If you use **Cursor**, **Claude Code**, or other coding agents on this repo, make sure they use [AGENTS.md](./AGENTS.md) or [CLAUDE.md](./CLAUDE.md).

## Start the quiz

**Product UI (React):** static BakeFile under `frontend/quiz/`. See
[`docs/FRONTEND_BACKEND.md`](./docs/FRONTEND_BACKEND.md) and
[`contracts/README.md`](./contracts/README.md).

```bash
cd frontend/quiz
npm install
npm run dev
```

Open <http://127.0.0.1:5173/>. Optional: copy
`contracts/examples/mini_bake.json` over `frontend/quiz/public/data/questions.json`
for a 4-question fixture.

Validate any BakeFile:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python tools/validate_quiz_bake.py
```

### Legacy Streamlit inspector (frozen)

The Streamlit question inspector is **frozen** — no new product features. Prefer
the React quiz above. For archaeological local inspection only:

```bash
.venv/bin/python tools/start_quiz.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\start_quiz.py
```

The legacy launcher opens <http://127.0.0.1:8501>. On a fresh clone it may copy a
bundled demo question into gitignored `data/`.

### First-time setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,inspector]"
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inspector]"
```

Requires Python 3.10+ and PyTorch 2.x.

### CPU and CUDA installation

The `pyproject.toml` dependency list is the package dependency source of
truth. The default `requirements.txt` is a CPU-oriented convenience entry
point used by Streamlit Community Cloud and other CPU-only deployments. For
local development, the editable install above uses the PyTorch wheel selected
by pip.

For CUDA, install a PyTorch build matching the installed NVIDIA driver and
the CUDA version supported by that build, then install this project:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -e ".[dev,inspector]"
```

Replace `cu121` with the CUDA wheel index required by the selected PyTorch release. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

Execution artifacts are device-specific. Do not mix CPU and CUDA candidate
artifacts in one candidate set or question, and do not treat their ground
truth values as directly interchangeable. `--device cuda` is strict: when
CUDA is unavailable, candidate generation fails explicitly instead of
silently falling back to CPU.

## Generate benchmark artifacts

Activate the virtual environment, then create datasets, candidate sets, and
questions through the interactive CLI:

```bash
source .venv/bin/activate

# PowerShell equivalent: .\.venv\Scripts\Activate.ps1

# Create a dataset
architecture-iq create-dataset -i

# Generate a candidate set
architecture-iq generate-candidates -i

# Assemble questions from one or more candidate sets
architecture-iq generate-question -i

# Evaluate an LLM on generated questions
python tools/llm_eval/run.py --model gpt-4o-mini
```

Artifacts are written under `data/` (gitignored).

## Dataset families (default profile `v1`)

| Family | Task | Models | Losses | Metric |
|--------|------|--------|--------|--------|
| `univariate_regression` | R → R symbolic regression | `mlp`, `kan` | MSE (+ L1/L2 reg) | `test_mse` |
| `multivariate_regression` | R^n → R symbolic regression | `mlp`, `kan` | MSE (+ L1/L2 reg) | `test_mse` |
| `bigram_lm` | Next-token prediction from fixed P(y\|x) | `transformer_lm`, `gru_lm` | cross-entropy (+ L1/L2 reg) | `test_ce` |
| `synthetic_tabular_classification` | Synthetic tabular binary classification (`xor`, `spiral`, and other rule families) | `mlp`, `kan` | cross-entropy | `test_ce` |

For `multivariate_regression`, **n** (input dimension) defaults to a random pick from the profile pool `input_dims: [2, 3, 4, 5, 8]`. Pin it with `--input-dim` or the interactive prompt.

Each family declares compatible model types; candidate sampling only draws from the intersection with `pools.model_types`. Config per family lives under `dataset_configs` in `profiles/v1.yaml`.

Default `v1` equally enables every registered dataset family and each family's compatible models. Older `v2*` pilot profiles remain for historical experiments; new question generation should use `v1` (or an explicit `--profile`).

## CLI reference

All commands accept `--profile NAME` (`v1` is the default). Run `architecture-iq --help` or `architecture-iq <command> --help` for details.

Interactive mode (`-i` / `--interactive`) prompts for every parameter. **Enter** on a choice field picks a random valid option. Interactive commands reject all other arguments except `--profile`. `generate-candidates -i` and `generate-question -i` only let you pick **existing** datasets (use `create-dataset` first).

### `create-dataset`

Create a new dataset instance under `data/datasets/{family}/{dataset_id}/`.

```bash
architecture-iq create-dataset --family univariate_regression --seed 42
architecture-iq create-dataset --random-family --seed 42
architecture-iq create-dataset --family multivariate_regression --seed 42 --input-dim 4
architecture-iq create-dataset -i
```


| Option                | Default               | Description                                              |
| --------------------- | --------------------- | -------------------------------------------------------- |
| `--seed`              | `0` (non-interactive) | Instance seed for dataset generation (see below)         |
| `--family`            | —                     | **Required** unless `--random-family` or `-i`            |
| `--random-family`     | off                   | Pick a random family from the profile pool               |
| `--input-dim`         | —                     | For `multivariate_regression` only: pin **n** (must be in profile `input_dims`) |
| `-i`, `--interactive` | off                   | Prompt for family, seed, and multivariate **n** (Enter = random from pool) |

**Multivariate input dimension.** By default, `multivariate_regression` samples **n** from the active profile's `dataset_configs.multivariate_regression.input_dims` (currently `[2, 3, 4, 5, 8]` in V1 and V2). Use `--input-dim` or `-i` to pin **n**; otherwise it is chosen randomly from that list (seeded by instance seed).

**What `--seed` controls:** the **instance seed** for synthetic data generation (expression formula and train/test point sampling). Same `--family` + same `--seed` reproduces the same dataset. It does **not** affect which family is picked when using `--random-family` (that draw uses a separate unseeded RNG).


### `generate-candidates`

Generate a named candidate set with ground truth. Each run writes candidates under `data/datasets/{family}/{dataset_id}/candidates/{set_name}/` where `set_name` looks like `set_{budget}_{model}_{optimizer}_{loss}_{hash}` (`var` or `fix` per axis).

```bash
architecture-iq generate-candidates data/datasets/univariate_regression/sym_XXXXXX \
  --budget 1024 --count 32 --vary model --vary optimizer --device cpu
architecture-iq generate-candidates -i
```


| Option                | Default                        | Description                                                  |
| --------------------- | ------------------------------ | ------------------------------------------------------------ |
| `dataset_path`        | **required** (non-interactive) | Path to dataset instance dir                                 |
| `--budget`            | **required** (non-interactive) | `total_samples_seen`                                         |
| `--count`             | **required** (non-interactive) | Number of candidates in this set                             |
| `--vary`              | **required** (non-interactive) | Repeat: `model`, `optimizer`, or `loss` (axes that may vary) |
| `--device`            | profile/default               | Execution device for new candidates: `cpu` or `cuda`        |
| `--seed`              | `0`                            | RNG seed for candidate sampling (see below)                    |
| `-i`, `--interactive` | off                            | Prompt for varying/invariant axes and fixed values           |

**What `--seed` controls:** the RNG for **sampling candidate specs**—which models/optimizers/losses are drawn on varying axes, and (in non-interactive mode) the random picks for invariant axes and batch size. It also salts the set directory name (`set_…_{hash}`). The seed is stored in `set.json`. It does **not** control ground-truth training seeds (those come from the profile's `base_seed` / `n_seeds`) or the dataset itself.

**Non-interactive vs `-i`.** Both modes use the same sampling and ground-truth pipeline. The difference is control over **invariant** axes (everything not listed in `--vary`, plus batch size, which never varies within a set):

- **Non-interactive:** invariant values are chosen **randomly once** per set, seeded by `--seed`. For example, `--vary model` fixes one optimizer and one loss for all candidates without prompting.
- **`-i`:** you are prompted to **pin** each invariant axis (model, optimizer, loss, batch size). **Enter** on a prompt accepts the same random sample non-interactive mode would use.

Interactive mode is not a different generator—it is strictly **more expressive** for pinning fixed components. There are no CLI flags today to pass those pins without `-i`.

### `generate-question`

Assemble one or more multiple-choice questions from candidate set(s) and write `prompt.txt`. Each invocation creates a run folder under `data/datasets/{family}/{instance}/questions/run_{n}q_{c}c_{hash}/` containing `run.json` and one directory per question. Question type is inferred automatically from the chosen candidates' specs.

```bash
architecture-iq generate-question data/datasets/univariate_regression/sym_XXXXXX \
  data/datasets/univariate_regression/sym_XXXXXX/candidates/set_1024_var_fix_fix_XXXXXX \
  --num-questions 5

# Multiple sets (e.g. different budgets or varying axes)
architecture-iq generate-question data/datasets/univariate_regression/sym_XXXXXX \
  data/datasets/.../candidates/set_1024_var_fix_fix_AAAAA \
  data/datasets/.../candidates/set_2048_fix_var_fix_BBBBB \
  --num-questions 3

architecture-iq generate-question -i
```


| Option                | Default                        | Description                                    |
| --------------------- | ------------------------------ | ---------------------------------------------- |
| `dataset_path`        | **required** (non-interactive) | Path to dataset instance dir                   |
| `candidate_sets`      | **required** (non-interactive) | One or more candidate set dirs                 |
| `--num-questions`     | **required** (non-interactive) | Questions to generate from the union pool      |
| `--num-choices`       | profile (`2`)                  | Choices per question (letters A, B, …)         |
| `--seed`              | `0`                            | RNG seed for question assembly (see below)     |
| `-i`, `--interactive` | off                            | Prompt for dataset, candidate sets, and counts |

**What `--seed` controls:** the RNG for **assembling questions from existing candidate set(s)** (no ground truth is re-run). Specifically:

- **Subset selection** — when more significant subsets exist than `--num-questions`, which ones are kept (order after shuffling the passing list).
- **Letter assignment** — shuffles which choice letter (A, B, …) each candidate gets; the significance winner stays correct but its letter may move.
- **Run folder name** — salts the `run_{n}q_{c}_{hash}` directory name.

The seed is stored in `run.json`. For typical pool sizes, subset search is exhaustive and deterministic aside from these shuffle steps. If the pool is very large, the seed also drives random combo sampling when exhaustive search is skipped. It does **not** change candidate metrics, which subsets pass significance, or which candidate is the correct answer.


## Typical workflows

### Architecture-only questions

Generate a set where only the model varies, then assemble questions.

```bash
architecture-iq create-dataset --family univariate_regression --seed 0
architecture-iq generate-candidates data/datasets/univariate_regression/sym_XXXXXX \
  --budget 1024 --count 32 --vary model
architecture-iq generate-question data/datasets/univariate_regression/sym_XXXXXX \
  data/datasets/univariate_regression/sym_XXXXXX/candidates/set_1024_var_fix_fix_XXXXXX \
  --num-questions 5
```

### Cross-budget mixed questions

Generate separate sets (e.g. different budgets or varying axes), then pass all set paths to `generate-question`.

```bash
architecture-iq create-dataset --family univariate_regression --seed 0
architecture-iq generate-candidates data/datasets/univariate_regression/sym_XXXXXX \
  --budget 1024 --count 32 --vary model --vary optimizer --vary loss
architecture-iq generate-candidates data/datasets/univariate_regression/sym_XXXXXX \
  --budget 2048 --count 32 --vary model --vary optimizer --vary loss
architecture-iq generate-question data/datasets/univariate_regression/sym_XXXXXX \
  data/datasets/.../candidates/set_1024_var_var_var_XXXXXX \
  data/datasets/.../candidates/set_2048_var_var_var_YYYYY \
  --num-questions 5
```

### Interactive session

```bash
architecture-iq create-dataset -i
architecture-iq generate-candidates -i
architecture-iq generate-question -i
```

## Layout

```
profiles/v1.yaml          # V1 profile (pools, grids, ground-truth settings)
AGENTS.md                 # AI agent development guide (canonical)
CLAUDE.md                 # Same as AGENTS.md (Claude Code)
docs/                     # Architecture, plans, reports, and release records
prompts/templates/        # NL prompt templates
src/architecture_iq/      # Pipeline: datasets, candidates, ground truth, questions
tools/llm_eval/           # Standalone LLM evaluation runner
tools/quiz_bundle/        # Canonical quiz bundle validation and publishing
tools/ranking_questions/  # Calibration-plus-ranking generation and scoring
tools/*analysis*.py       # Offline curve/order-parameter analysis
templates/                # Reusable HTML report template
data/                     # Generated datasets, candidates, questions (runtime)
examples/quiz_demo/bundle/ # Version-controlled deployed question bundle + manifest
llm_runs/                 # LLM evaluation runs (runtime)
```

## Reproducibility

Ground truth **executes the on-disk Python files**, not parallel framework shortcuts:

- **Datasets:** `synthesize.py` is loaded via `importlib` and `synthesize()` materializes `train.pt` / `test.pt`.
- **Candidates:** `train.py` imports `model.py`, `optimizer.py`, and `loss.py` from the same folder. The runner calls `train_and_eval()` in `train.py`. Before each run, `.py` files are regenerated from `candidate_spec.json` so specs and code stay aligned.

See `src/architecture_iq/runtime/loader.py`.

## Question inspector (frozen)

> **Frozen.** Do not add product features here. The public/internal quiz path is
> `frontend/quiz/` + BakeFile (`contracts/`). See
> [`docs/FRONTEND_BACKEND.md`](./docs/FRONTEND_BACKEND.md).

Streamlit UI for browsing and taking questions. Original benchmark artifacts remain
read-only; user-created training settings live in a per-browser-session temporary
directory and are cleared when the user switches questions.

While solving, expand **Add custom setting** to choose the architecture, optimizer,
loss, training budget, and seed count. Confirming the form trains that setting on the
current dataset and adds its learning curve to the page. A custom setting can inherit
all editable values from Choice A/B/C. The inspector retains at most two custom runs:
the newest run and the historical run with the lowest final loss.

Answers, proposed settings, completed custom runs, post-result surprise
reactions, question-presentation decisions, and per-question comments can be
collected in a session trace. The sidebar can download the full trace as JSON
and upload its pending events when both the feedback endpoint and Bearer token
are configured; comments and surprise reactions use the same endpoint as
one-event uploads. Uploads are
receiver-sized and a conflicting ID is quarantined so it cannot block later
events. See the inspector README for the endpoint contract and configuration,
and [supabase/README.md](./supabase/README.md) for the deployable receiver and
report views. The Supabase receiver acknowledges an unchanged event-ID replay as
an idempotent duplicate, but rejects reuse of that ID for different logical
content with HTTP 409; a conflicting batch is not partially inserted.
Feedback JSON is fail-closed to the Python/JavaScript/PostgreSQL lossless subset:
integer-valued numbers must stay within ±(2^53−1), strings cannot contain
unpaired Unicode surrogates, and identifier/comment limits count Unicode code
points so emoji are handled consistently on both client and receiver.

With no explicit local question path, the inspector reads the version-controlled
quiz bundle directly and attests its release hash plus every artifact path,
size, and SHA-256 before serving questions. A missing, stale, forged, or tampered
bundled release fails closed rather than being mislabeled as the deployed quiz.

For the current default bundle, that attested pool contains exactly 60 published
questions. **Next** now builds a private cold-start catalog from only those 60
manifest entries and their stored question/spec/summary artifacts; it never
scans additional local questions or reruns training. Validity is a hard gate.
Tie-aware model-size/depth/width and optimizer shortcuts seed a Beta posterior,
then the local policy uses 80% exploitation and 20% minimum-exposure exploration,
excludes already completed/current questions, and avoids repeating a family when
another family is available. A catalog/policy error falls back to sequential
Next. The selected question, policy version, random decision ID, mode, exact
mixture propensity, source, and session position are recorded locally as a
`question_presented` event. The catalog and its scores remain private and never
modify or leak the answer key, GT, prompt, or release artifacts.

This first policy uses the offline cold-start prior and presentation counts from
the current local attempt. It does not yet fetch hosted reaction aggregates or
claim cross-session personalization or an A/B improvement.

Restoring upload from a previously downloaded session JSON is implemented in the
local inspector. It strictly validates the complete file before any network
request and places its events into an outbox separate from the live trace. The
file cannot import browser-local acknowledged or quarantined state from an
earlier session; within the current browser session, a canonical full-content
recovery ID preserves progress so retrying the same file sends only pending
events.

Deployment status (2026-07-12): the upload/outbox, downloaded-trace recovery,
runtime attestation, surprise reaction/catalog/recommended-Next behavior,
registry, and authoritative Reports behavior above is
implemented and tested only in the current local working tree. It has not
completed a real hosted receiver acceptance run. A fresh anonymous browser check
of <https://architecture-iq.streamlit.app/> loads the 60-question inspector and
its custom-setting UI, but shows neither the session-upload control nor the
single-question comment control. Its Streamlit creator link and this repository's
Git remote both identify `renrua52`; the exact deployed source commit still needs
Streamlit Cloud console or runtime-SHA evidence. The new local inspector can
read the actual Git checkout commit and cross-check it against allowlisted
platform declarations, but that capability is not present on the currently
hosted old revision. The remote `main` commit lacks these uncommitted local
feedback features.

The startup command at the top opens the default question. To open a specific
question or question run first, pass its path to the launcher:

```bash
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX/q_XXXXXX
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX
```

See [tools/question_inspector/README.md](./tools/question_inspector/README.md).

## Internal feedback reports

Uploaded feedback has a separate, read-only internal Streamlit entrypoint:

```bash
.venv/bin/python -m streamlit run tools/feedback_reports/app.py
```

It provides six business tabs—Summary, Sessions, Questions, Answers, Proposals,
and Comments—plus Ingestion observability, Registry quality, Surprise, and
derived Data quality, for ten tabs total. The protected endpoint allowlists
thirteen RPCs; the
authority-status and exact-event-resolution RPCs are verifier/operator surfaces,
not dashboard tabs. The app does not read the quiz browser session directly.
Ingestion metrics
separate verified idempotent retries, legacy unclassified duplicates, and real
event-ID content conflicts. Rates cover only persisted,
authenticated POST outcomes and are explicitly not an end-to-end network
success rate.
The local STATS-003 implementation registers immutable release/question/choice
facts from the attested bundle. Business dimensions and answer correctness are
derived only from exact release + question ID + question-version membership;
missing or mismatched raw events remain auditable in Registry quality and never
fall back to client-reported `is_correct`, family, dataset, or question type.
The local REPORT-002 implementation adds paginated, registry-matched answer and
proposed/rejected-setting rows with the same authoritative filters, strict
response schemas, safe JSON display, and complete-page-only CSV export. Forward
migration `18000` appends global `session_id` and `attempt_id` filters to the
historical positional signatures of all six business RPCs and the snapshot,
preserving callers that omit them.
Forward migration `19000` keeps the event wire schema at `1.0` while adding
strict `question_presented` and `question_reaction_submitted` enum/payload
contracts to both the table and the atomic ingest RPC. The Inspector only offers
**Surprised / As expected** after answer reveal; the first response for an
attempt has a deterministic event ID and remains separate from correctness,
comments, and future likes. Presentation events carry the release/attempt,
decision and policy IDs, mode, propensity, source, and position used to evaluate
navigation rather than infer exposure from answers.
Forward migration `20000` locally adds two service-role-only SURPRISE-002 RPCs.
The per-question RPC counts the first valid post-answer rating for each exact
session/attempt/release/question/version, reports yes/no counts, rating coverage,
observed surprise, and a Beta(1,1) posterior mean. Its quality companion keeps
raw/valid/orphan/duplicate counts and a conserved orphan breakdown. The local
Reports UI exposes both on the Surprise tab with the same eight identity/time
filters. These two calls are independent of the six-page business MVCC snapshot
and of each other, and their values are not yet fed back into Inspector Next.
The local STATS-004 implementation replaces the six-request browser fanout
with one business-snapshot GET. The Edge Function makes one PostgREST call to
the `18000` recreation of the `17000` SQL function, which forwards both identity
filters and assembles all six pages in one PostgreSQL
statement and therefore one MVCC snapshot. Its `business_snapshot_v1` row
includes server `snapshot_at`, the embedded `registry_v1`/`detail_v1` authority
facts and registry counts, plus the six page envelopes as strict `pages_json`
text. Exact totals still scan all matches, but only the requested stable row
prefix reaches JSON serialization and per-page UTF-8 byte budgeting; rows are
never field-truncated, and the final `pages_json` is capped at 4 MiB. The Edge
validates then forwards the raw PostgREST snapshot array so PostgreSQL bigint
authority counts do not round-trip through JavaScript `Number`. A byte- or
row-truncated page retains its exact total and cannot be exported as a complete
CSV. The app atomically replaces its saved business result only after the whole
response validates; a failed refresh keeps the previous complete snapshot.
Ingestion observability and Registry quality remain independent, non-atomic
requests and are unavailable while any release/family/type/question/session/
attempt filter is active. Authority status accepts no filters, while exact-event
resolution accepts only `event_id`; neither auxiliary verifier surface accepts
identity filters.
The implementation is available locally but is not a hosted deployment. The
entrypoint has no built-in user login, so any hosted instance must be a
separate Streamlit app with platform-level access restricted to maintainers.
See
[tools/feedback_reports/README.md](./tools/feedback_reports/README.md) for the
UI/filter/CSV contract and [supabase/README.md](./supabase/README.md) for the
separate report token, migration, Edge Function deployment, and opt-in hosted
write/read roundtrip verifier. The same guide includes an environment-only,
rollback-only PostgreSQL staging verifier that checks the real migration
catalog, ACL/RLS/trigger/constraint posture and recomputes the exact current
registry hash before endpoint smoke. Neither verifier has been run against a
real hosted revision yet. The endpoint verifier proves the normal single-event
fresh/resume path, a three-event answer/proposal/comment batch followed by its
unchanged idempotent replay, and, by default, a real same-ID/different-content
409. Its CLI attests a published bundle, chooses a real registered membership,
deliberately
sends wrong client dimensions and inverse answer correctness, then requires the
exact-event report to return registry-derived identity/correctness and mismatch
flags. A dedicated exact seven-column authority-status row requires both the
`registry_v1` aggregate cutover and `detail_v1` answer/proposal cutover. Before
the first write, the verifier also requires both detail RPCs to pass protected
empty-page negative controls and requires one fully empty, authority-attested
six-view snapshot for a random nonexistent question. A successful run reports
`business_snapshot_verified=true`, then validates the uploaded answer and
proposal as exact authoritative detail rows. The verifier does not rely on
shared aggregate counts, so historical answers for the selected question are
safe. It then requires the real session/attempt pair to produce the exact six
business pages and separately requires an incorrect session and incorrect
attempt to produce empty snapshots. Only that complete positive/negative proof
sets `session_attempt_filters_verified=true`. The Reports UI does not make a
separate authority-status request: its
strict snapshot parser requires the embedded revisions, booleans, counts, and
six complete page contracts before rendering. Explicit skip flags are reserved
for compatibility, minimal-footprint, or interrupted recovery and leave their
verification fields false.

## Ranking questions and analysis tools

Generate calibration-plus-ranking tasks, create de-identified agent bundles,
and score predicted orders by inversion count:

```bash
python tools/ranking_questions/generate.py <candidate-set> --layout anchored
python tools/ranking_questions/make_blind_bundle.py <run> <public-bundle> \
  --answer-key-output <private-key.json>
python tools/ranking_questions/score_answers.py <run> <answers.json>
```

Additional offline utilities:

- `tools/analyze_order_parameters.py` summarizes trained candidate curves.
- `tools/evaluate_arithmetic_rules.py` evaluates deterministic selection rules.
- `tools/make_single_question_blind_quiz.py` builds isolated single-question bundles.
- `tools/build_readme_case_assets.py` rebuilds the case-study data used by
  [README.html](./README.html).

Generated analysis and ranking runs under `artifacts/` are gitignored. See
[tools/ranking_questions/README.md](./tools/ranking_questions/README.md) for the
ranking workflow and leakage precautions.

## LLM evaluation

Standalone runner that sends question prompts to an OpenAI-compatible chat API, parses the model's letter answer from an `<answer>` tag, and scores against ground truth. Does not import `architecture_iq`; reads question artifacts under `data/` (dataset-scoped runs and legacy `data/questions/`).

Prompts are augmented at eval time so the model can reason freely, then commit with e.g. `<answer>B</answer>`. The full raw response is stored before parsing.

Set API credentials (any OpenAI-compatible host):

```bash
export OPENAI_API_BASE="..."
export OPENAI_API_KEY="sk-..."
```

Run over all questions under `data/` (default), a question run folder, or legacy `data/questions/`:

```bash
python tools/llm_eval/run.py --model gpt-4o-mini
python tools/llm_eval/run.py data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX --model gpt-4o-mini
python tools/llm_eval/run.py --model gpt-4o-mini --temperature 0 --limit 10 --workers 8
```

Each run writes under `llm_runs/{timestamp}_{model}/`:

- `run.json` — model config and accuracy summary
- `results/{question_id}.json` — per-question ground truth, parsed letter, raw response, and chain-of-thought (when present)

Use `--run-dir path/to/run --skip-existing` to resume a partial run.

See [tools/llm_eval/README.md](./tools/llm_eval/README.md).

### Context-protocol boundary

The historical local revealed-answer runner and its old `llm_runs/` artifacts
are reproducibility references only. They append compact prior answer/outcome
lines to later API prompts and should not be used as the current `main` context
benchmark.

The canonical current feedback protocol is
`tools/sequential_feedback_session.py`: the agent must submit a prediction
before feedback is returned, then receives the correct candidate and per-choice
metric values and records a lesson. Its score is feedback-conditioned online
adaptation, not a replacement for blind/static capability scores. Always report
the exact protocol and question scope when comparing results.

## Tests

```bash
pytest
```
