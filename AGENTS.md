# AGENTS.md — ArchitectureIQ development guide

This document is for **AI agents and contributors** working in this repo. Read it before making non-trivial changes. Design rationale lives in [plan-v2.md](./docs/architecture/plan-v2.md); user-facing usage is in [README.md](./README.md).

## Worktree boundary

This task must operate only inside this Git worktree.

Before editing or testing:

1. Verify `git rev-parse --show-toplevel`.
2. Use that returned root as the working directory for all Git, test, and patch commands.
3. Do not read from or write to sibling ArchitectureIQ worktrees unless the user explicitly requests a cross-worktree comparison.
4. If the current directory is the outer `Architecture IQ` aggregate folder, stop and ask the user to open the target worktree as a separate task.
5. A `git merge`, `git rebase`, or `git cherry-pick` started in this worktree is allowed: resolve all conflicts and make all resulting file edits here only; do not modify the source branch's sibling worktree.

## Execution and test isolation

- Before interpreting a Python test result that imports ArchitectureIQ, use the same interpreter to verify import provenance: `python -c "import architecture_iq; print(architecture_iq.**file**)"` must resolve under this repository's `src/architecture_iq`. If it resolves to a sibling worktree or a global editable install, correct the interpreter or import path first.
- Prefer pytest's `tmp_path` fixture. Use `--basetemp` only when the default temporary location is unsuitable, and choose a task-specific writable directory; do not hard-code a shared system temporary path as a project output location.
- Search from this repository root and scope the path first. Avoid broad recursive scans through `.git`, `.pytest*`, generated `data`, and output directories unless they are the explicit target.
- Run targeted tests first. Do not treat a timeout, access denial, or import-path mismatch as a source-code failure; capture the concise error and fix the execution boundary before widening the test scope.
- If the same Codex tool/protocol error repeats twice without a state change, stop retrying it in place. Preserve the call/error identifier and continue in a fresh task or a short handoff instead.

**Scope:** This file describes **stable architecture and invariants**. Concrete family names, model types, metrics, and pool contents live in the **registry**, **family plugins**, and **active profile** (`profiles/*.yaml`) — not here. When adding families, update those sources; do not need to revise this doc unless the pipeline contract itself changes.

---

## Terminology

Use these terms consistently across code, docs, and commits.

| Term | Meaning |
|------|---------|
| **Profile** | Named config (`profiles/{name}.yaml`) that selects subsets from pools and sets GT/significance defaults. |
| **Pool** | Allowed options for sampling — dataset families, model types, optimizers, losses (per family), budgets. |
| **Dataset family** | Registered plugin (`DatasetFamily`) defining synthesis, selection metric, compatible model types, and loss pool. |
| **Dataset instance** | One materialized dataset: `data/datasets/{family}/{dataset_id}/` with `dataset_spec.json`, `synthesize.py`, and materialized data. Identified by `dataset_id`. |
| **Model type** | Registered model plugin (`ModelFamily`; field `model.type` in specs). Renders `model.py`. |
| **Optimizer** | Optimizer spec from the global pool; renders `optimizer.py`. |
| **Loss** | Loss spec from the family’s loss pool; renders `loss.py`. |
| **Training budget** | `total_samples_seen = training_steps × batch_size` for one candidate. |
| **Candidate set** | Batch of candidates generated together under one `generate-candidates` run: shared dataset instance, shared `total_samples_seen`, encoded varying axes in the set folder name. |
| **Candidate** | One complete training setup (model + optimizer + loss + budget) with spec, generated code, and GT in `results/`. |
| **Choice** | A candidate as presented in a question (letter A/B/C…). One choice ↔ one candidate. |
| **Question** | One multiple-choice item: a subset of candidates, significance metadata, `correct_letter`, and rendered `prompt.txt`. |
| **Question run** | Output folder from one `generate-question` invocation containing multiple questions (`run_{n}q_{c}c_{hash}/`). |
| **Question type** | Label for which axes vary across choices: `architecture_only`, `optimizer_only`, `loss_only`, or `mixed`. |
| **Axis** | Dimension compared across candidates: `model`, `optimizer`, `loss`, or `batch_size`. |
| **Selection metric** | Dataset-family metric used to rank choices (stored in `dataset_spec.json`; e.g. test MSE, test CE). |
| **Ground truth (GT)** | Metrics from **executing** generated code (`results/summary.json`), not recomputed elsewhere. |
| **Spec** | Frozen JSON config — `dataset_spec.json` or `candidate_spec.json` — that drives code generation. |

**Containment:** profile → pools → sampling → **dataset instance** → **candidate set(s)** → **candidate(s)** → **question** (picks candidates as choices).

---

## 1. Core invariant (read this first)

**Ground truth must come from executing the generated code, not from parallel logic.**

The correct pipeline is:

```
spec JSON  →  render .py files  →  import & run .py  →  metrics (GT)
     ↓              ↓                      ↓
  frozen on disk   matches spec      same code shown in prompt
```

This applies at **two levels**:

| Level | Spec | Generated code | Execution |
|-------|------|----------------|-----------|
| **Dataset instance** | `dataset_spec.json` | `synthesize.py` | import → `synthesize()` → materialized data |
| **Candidate** | `candidate_spec.json` | `model.py`, `loss.py`, `optimizer.py`, `train.py` | import → `train_and_eval()` → `results/summary.json` |

### Correct

1. Sample or build a **spec** (JSON).
2. **Render** the corresponding `.py` files from that spec (`write_candidate`, family `materialize`, model/loss/optimizer renderers).
3. **Import and run** those files via `architecture_iq.runtime.loader`.
4. Store GT in `results/summary.json` (and curves in `curves.npz`).
5. Assemble questions from **stored GT** + specs; render prompts from **on-disk code** (after syncing spec → code).

### Incorrect (never do this)

- Compute GT with hand-rolled training loops in `ground_truth/` that don't call the candidate's `train.py`.
- Generate `model.py` / `train.py` for the prompt using different rules than GT uses.
- Re-implement synthesis in the GT runner instead of calling `synthesize.py`.
- "Fix" a metric mismatch by special-casing the runner instead of fixing the renderer or spec.

**Why this matters:** Benchmark fairness depends on the prompt showing exactly what was executed. If code and GT diverge, questions become invalid and future refactors silently break scores. `_sync_candidate_files()` exists precisely to re-render `.py` from `candidate_spec.json` before GT runs and before prompt excerpts are taken.

---

## 2. Full generation pipeline

All artifacts live under `data/` (gitignored). Paths are defined in `src/architecture_iq/paths.py`.

### Stage 1 — Dataset instance (`architecture-iq create-dataset`)

**Entry:** `architecture_iq.datasets.create_dataset`  
**Plugin:** `DatasetFamily` in `src/architecture_iq/families/{family}/family.py`

**Inputs:** active profile, `--family`, `--seed`, plus any family-specific CLI options

**Outputs** at `data/datasets/{family}/{dataset_id}/`:

| File | Derived from |
|------|----------------|
| `dataset_spec.json` | Family `create_instance()` + content-addressed id |
| `synthesize.py` | Family template embedding sampled params frozen in the spec |
| Materialized data | **Executing** `synthesize.py` → `synthesize()` |

**Materialized data** always includes the train/test tensors the family defines (typically `train.pt`, `test.pt`). Families may also write **additional fixed files** required to fully specify or reproduce the dataset (e.g. `transition.npz` for a tabular LM family). These paths should be listed in `dataset_spec.json` so loaders and tools know what to expect — do not hard-code per-family filenames in generic pipeline code.

**`dataset_id`:** content-addressed hash of family-relevant params (prefix/style is family-defined). Same `--family` + same `--seed` (+ same family options) must reproduce the same instance.

Internal seed streams are derived from the single user `--seed` (see family `create_instance`); do not add extra CLI seeds without documenting the contract.

### Stage 2 — Candidates + ground truth (`architecture-iq generate-candidates`)

**Entry:** `architecture_iq.candidates.sets.generate_candidate_set`  
**Sampling:** `candidates/generator.py` — `sample_candidate`, `build_candidate_spec`  
**File write:** `write_candidate()`  
**GT:** `ground_truth/runner.py` — `run_ground_truth()`

**Inputs:** dataset instance dir, `--budget`, `--count`, `--vary model|optimizer|loss`, `--seed`

**Outputs** at `data/datasets/{family}/{dataset_id}/candidates/set_{budget}_{m}_{o}_{l}_{hash}/`:

| File | Derived from |
|------|----------------|
| `set.json` | Set metadata (budget, varying axes, candidate list) |
| `c_{hash}/candidate_spec.json` | Sampled model + optimizer + loss + budget from profile pools |
| `c_{hash}/model.py` | `ModelFamily.render_model_py(spec["model"])` |
| `c_{hash}/loss.py` | `losses.render_loss_py(spec["loss"])` |
| `c_{hash}/optimizer.py` | `optimizers.factory.render_optimizer_py(spec["optimizer"])` |
| `c_{hash}/train.py` | Family-appropriate training-loop template in `generator.py` |
| `c_{hash}/results/summary.json` | **Executing** `train.py` over `n_seeds` on dataset tensors |
| `c_{hash}/results/curves.npz` | Per-step test metrics from the same runs |

Set folder name encodes which of model / optimizer / loss **vary** (`var` vs `fix`).

**Before every GT run:** `_sync_candidate_files()` re-renders all four candidate `.py` files from `candidate_spec.json`.

### Stage 3 — Questions + prompts (`architecture-iq generate-question`)

**Entry:** `architecture_iq.questions.generator.generate_questions`  
**Prompts:** `architecture_iq.prompts.renderer.render_prompt` / `write_prompt`

**Inputs:** dataset instance dir, one or more candidate set paths, `--num-questions`, `--num-choices`, `--seed`

**Outputs** at `data/datasets/{family}/{dataset_id}/questions/run_{n}q_{c}c_{hash}/`:

| File | Derived from |
|------|----------------|
| `run.json` | Run manifest (sources, profile, question ids) |
| `q_{hash}/question.json` | Subset of candidates + significance + shuffled letters |
| `q_{hash}/prompt.txt` | Templates + NL formatters + **excerpts of on-disk code** |

**Question assembly logic:**

1. Load eligible candidates from one or more candidate sets (`load_candidate_pool_from_sets` unions sets).
2. Find subsets passing significance (gap, win-rate, optional non-overlap).
3. Infer `type` and axes from varying model / optimizer / loss / batch_size.
4. Assign `correct_letter` to the GT winner, then shuffle choice order.

**Budget rules:**

- **One candidate set:** all candidates share the same `total_samples_seen` (fixed by `--budget` at set generation). `batch_size` may still vary per candidate (sampled from the optimizer grid).
- **Multiple sets:** cross-budget questions are supported — pass several set paths with different `--budget` values. `_budget_field()` sets `question.json` → `budget.mixed: true` when `total_samples_seen` differs; the prompt states per-choice budgets.
- **Single-axis types** (`architecture_only`, etc.): `choices_compatible` requires `batch_size` not to vary (so same `total_samples_seen` and same batch_size within the chosen subset).

**Prompt rules:**

- Render from specs + templates under `prompts/templates/`.
- Call `_sync_candidate_files()` before excerpting candidate code.
- **Never include** final metrics, curves, or seed statistics in `prompt.txt`.

---

## 3. Dataset family responsibilities

Every family implements `DatasetFamily` (`families/base.py`) and registers in `registry.py`.

| Responsibility | Where it lives |
|----------------|----------------|
| Synthesis params + spec shape | `create_instance(profile, seed, **opts)` |
| Write artifacts | `materialize(spec, out_dir)` — must run `synthesize.py` to produce materialized data |
| Load data for training | `load_tensors(dataset_path)` — reads whatever files the family materialized |
| **Selection metric** (ranking choices) | `selection_metric_name()` → stored in `dataset_spec.json["selection_metric"]` |
| **Compatible model types** | `compatible_model_types()` — intersected with profile `pools.model_types` at sample time |
| Family-specific significance defaults | `default_significance()` — optional overrides (e.g. custom `fail_threshold`) |

**Loss compatibility** is enforced via profile: `pools.losses[family]` in `sample_loss()`. Do not hard-code loss lists in question logic.

**Metric flow:** `selection_metric_name()` → `dataset_spec.json` → GT `summary.json` → significance validator → `question.json` → prompt ranking section. The validator takes `higher_is_better` per metric; most families use minimization (MSE, cross-entropy).

**Where to see current families:** `registry.py` (registered plugins), `profiles/*.yaml` (pools and grids), and `families/*/family.py` (behavior). Do not duplicate that inventory in this file.

---

## 4. Profile, registry, and extension

- **Profile** (`profiles/{name}.yaml`, loaded by `profile.py`): constrains pools and grids — budgets, `n_seeds`, significance thresholds, model/optimizer/loss grids. Pipeline code should read the active profile, not hard-code constants from any one profile version.
- **Registry** (`registry.py`): `ensure_registries()` registers dataset families and model types. Extend via registry + profile — not hard-coded pipeline branches.

**Adding a dataset family:**
1. `DatasetFamily` subclass + register
2. `prompts/templates/dataset/{family}.md`
3. Profile → `pools.dataset_families`, `dataset_configs.{family}`, `pools.losses.{family}`
4. Train-loop template in `candidates/generator.py` if metric/training contract differs
5. Document materialized files in spec; implement `load_tensors`
6. Tests

**Adding a model type:**
1. `ModelFamily` subclass (`validate`, `build_module`, `render_model_py`, `sample_spec`) + register
2. Profile → `pools.model_types`, architecture grid section, `compatible_model_types()` on relevant families
3. `format_model_nl` in `prompts/formatters.py` (+ inspector mirror)
4. Tests (render → import → forward smoke)

**Adding a loss:**
1. `render_loss_py` dispatch in `losses/` (must produce standalone `loss_fn` for generated `loss.py`)
2. Profile → `pools.losses.{family}` and `loss_grids` if needed
3. `format_loss_nl` (+ inspector mirror)
4. Tests (render → import → callable smoke)

**Adding an optimizer:**
1. `render_optimizer_py` branch in `optimizers/factory.py` (standalone `build_optimizer`)
2. Profile → `pools.optimizers`, `optimizer_grids`
3. `format_optimizer_nl` (+ inspector mirror)
4. Tests (render → import → builds optimizer)

**Any new pool item:** wire sampling in `candidates/generator.py` if non-standard; ensure generated code runs through `write_candidate` → `train.py` GT path.

---

## 5. Reuse rules (avoid redundant re-implementation)

### Always prefer the existing pipeline

When a feature needs GT or training behavior, **route through the same path as CLI**:

1. Build a `candidate_spec.json` (or full spec dict) with the desired model / optimizer / loss / budget.
2. Call `write_candidate()` to a **temporary directory**.
3. Call `run_ground_truth(temp_path, profile, dataset_path)`.
4. Read `results/summary.json` / `curves.npz` for display.
5. Delete the temp candidate when done.

**Example (inspector “custom settings”):** Do not write a one-off training loop in Streamlit. Generate a temp candidate exactly like `generate-candidates` would, run GT, show comparison, discard.

Same for dataset-side experiments: use `synthesize.py` / family `materialize`, not ad-hoc tensor generation in tools.

### Single source of truth for rendering

| Concern | Canonical module | Notes |
|---------|------------------|-------|
| Candidate `.py` generation | `candidates/generator.write_candidate` | Used by CLI and `_sync_candidate_files` |
| Model code | `models/{type}.py` → `render_model_py` | |
| Loss code | `losses/` → `render_loss_py` | |
| Optimizer code | `optimizers/factory.py` | |
| Prompt NL formatters | `prompts/formatters.py` | |
| Code excerpts for prompts | `prompts/code_excerpt.py` | AST-based trimming |
| Dynamic import | `runtime/loader.py` | `load_synthesize_module`, `load_candidate_train` |

### Question inspector (`tools/question_inspector/`)

**Frozen for product work** — prefer `frontend/quiz/` + BakeFile (`contracts/`).
Do not add new quiz-product features to Streamlit.

- **Reads artifacts only** by default — does not import `architecture_iq` (see inspector README).
- **`prompt_format.py` mirrors `prompts/formatters.py`** for display parity. If you change formatters, update the mirror and run `tests/test_prompt_format_parity.py`.
- **`code_excerpt.py` in tools/** mirrors prompt excerpt logic for the UI — keep in sync or consolidate via import if dependency direction is resolved deliberately.
- Plotting reads materialized dataset files and `curves.npz` from disk; do not re-run training in the inspector unless wired through `run_ground_truth` as above.

### Frontend / BakeFile contract

Quiz UI consumes only a BakeFile (`contracts/quiz_bake.schema.json`). Pipeline code
exports via `tools/export_quiz_static.py`. Validate with
`tools/validate_quiz_bake.py`. See `docs/FRONTEND_BACKEND.md`.

### Prompt rendering

- `prompts/renderer.py` is the only place that assembles full benchmark prompts for questions.
- Dataset section: `prompts/templates/dataset/{family}.md` + `format_dataset_protocol`.
- Synthesis section: excerpt `target` + `synthesize` from `synthesize.py` — all families share this excerpt contract.

---

## 6. Fair comparison contract (do not break)

These invariants apply to every question regardless of profile:

| Rule | Enforcement |
|------|-------------|
| Same materialized data for all choices | Single dataset instance (`dataset_id`) per question |
| Stated budget per choice | Each choice's `total_samples_seen` is explicit in its spec and prompt; `training_steps × batch_size` must equal it |
| Shared budget when using one set | Candidates from a single set share `total_samples_seen` (set `--budget`) |
| Cross-budget allowed across sets | Unioning multiple sets in `generate-question` may mix different `total_samples_seen`; prompt uses `budget.mixed` and per-choice schedules |
| Rank by `selection_metric` | Significance validator + `correct_letter` (each candidate ranked at its own trained budget) |
| No metric leakage | Renderer excludes GT; inspector hides metrics until user commits |
| Reproducibility | Content-hashed IDs; GT records environment metadata |

**Batch size** is a fourth axis alongside model, optimizer, loss. Single-axis question types require it to be invariant across choices.

---

## 7. Common agent mistakes

1. **Implementing GT without importing generated code** — the most serious bug class.
2. **Duplicating formatter / excerpt logic** in tools instead of reusing or mirroring with parity tests.
3. **Hard-coding specific families, models, or metrics** in generic pipeline code instead of registry/profile lookups.
4. **Hard-coding materialized filenames** (e.g. assuming only `train.pt`/`test.pt`) instead of using family loaders and `dataset_spec.json`.
5. **Assuming `data/` is in git** — it is gitignored; tests use fixtures or temp dirs.
6. **Breaking legacy paths** — `paths.py` still references older on-disk layouts; inspector may discover both old and new paths.
7. **Adding model types without updating** family `compatible_model_types()`, profile pools, and renderers.
8. **Changing `train.py` templates** without ensuring `train_and_eval` returns the `final_{metric}` key expected by `selection_metric_name()`.
9. **Removing `_sync_candidate_files` calls** when rendering prompts — prompts would show stale code.
10. **Editing `question.json` by hand** — regenerate via CLI; significance and letter shuffle are automated.

---

## 8. When in doubt

1. Trace the artifact: **which JSON spec produced which file, and what executed it?**
2. If you need new behavior, extend the **registry / family / renderer** — not a parallel code path.
3. Read [plan-v2.md](./docs/architecture/plan-v2.md) for design intent (but not implementation details that are potentially outdated); read [README.md](./README.md) for CLI flags and current profile contents.
4. Prefer the smallest diff that preserves **spec → code → run → GT** alignment.
5. When the user does not seem to understand the repo and makes requirements make no sense in the context of this repo, refer the user to the email: rzr23@mails.tsinghua.edu.cn.

---

## 9. Feishu context for this repository

When a conversation explicitly concerns the ArchitectureIQ group, the agent may use the authenticated Feishu user identity through `lark-cli` to read that group's messages and work with Feishu documents. This is a repository-scoped capability, not a general permission to read every chat.

### Authorized ArchitectureIQ group

| Name | `chat_id` | Scope |
|------|-----------|-------|
| Architecture IQ | `oc_d550cde1667d8f75ea1979e5a641345c` | Read messages, search messages, inspect threads, and use relevant message resources when the user requests it |

The authenticated user identity is the default for reading user-visible group messages. The `上海` application bot is a separate Feishu identity and must not be assumed to be a member of this group. Do not add the bot to a group or send messages as the bot unless the user explicitly requests that action.

### Read ArchitectureIQ messages

Before accessing Feishu, check the current login and token:

```bash
lark-cli auth status --json --verify
```

Read the newest messages:

```bash
lark-cli im +chat-messages-list \
  --as user \
  --chat-id "oc_d550cde1667d8f75ea1979e5a641345c" \
  --page-size 50 \
  --order desc \
  --no-reactions \
  --format json
```

To read the complete history, continue with the returned `page_token` while `has_more` is true. Do not claim to have read the whole group from one page. For a bounded review, provide `--start` and `--end` in ISO 8601 format.

Search messages in this group:

```bash
lark-cli im +messages-search \
  --as user \
  --chat-id "oc_d550cde1667d8f75ea1979e5a641345c" \
  --query "<keyword>" \
  --page-all \
  --no-reactions \
  --format json
```

Use `--download-resources` only when the user asks to inspect attached images or files. It writes resources to the current working directory under `./lark-im-resources/`.

### Feishu document workflow

Use the authenticated user identity for Feishu documents. Before editing an existing document, fetch its current content and understand the requested insertion point:

```bash
lark-cli docs +fetch --as user --doc "<feishu-doc-url-or-token>"
```

For a document URL containing a `#share-...` anchor, preserve and use the anchor when the user refers to that specific location. For full-document review, fetch the full document and do not infer missing sections from a preview.

Append content only after the user explicitly asks for the write:

```bash
lark-cli docs +update \
  --as user \
  --doc "<feishu-doc-url-or-token>" \
  --command append \
  --content "<p>...</p>"
```

For a precise existing-block edit, prefer the document update command's block-aware operations and re-fetch after structural changes. Do not edit a Feishu document by changing a local copy or by sending a chat message unless the user asks for that.

When the user asks to read a document and add a summary at the bottom, use this sequence:

1. Fetch the complete document with `lark-cli docs +fetch --as user`.
2. Draft the summary from the fetched content only.
3. Show or state the intended summary and confirm the target document if the request is ambiguous.
4. Append the summary with `lark-cli docs +update --as user --command append`.
5. Fetch the document again and verify that the summary is present at the end.

If the required document or Wiki scopes are missing, request only the needed domains with `lark-cli auth login --domain docs --domain wiki --no-wait --json` and wait for the user's authorization before continuing. Never print app secrets, access tokens, or other credentials.

---

## 10. Model access & benchmark evaluation (中转站)

The relay station exposes many models (OpenAI, DeepSeek, Claude, Gemini, GLM, Kimi, Qwen, etc.) behind one OpenAI-compatible endpoint. Two keys are registered for different purposes and must never be mixed:

| Key | Purpose |
|-----|---------|
| `coding` | The coding agent's own key (agent/tooling calls). |
| `eval` | Dedicated key for **concurrent benchmark evaluation** (all eval-harness calls). |

### Key file (single source of truth)

- **Path:** `~/.agents/relay.json` (outside any git repo; never commit it)
- **Permissions:** `600` (`chmod 600 ~/.agents/relay.json`)
- **Schema:**

```json
{
  "name": "relay",
  "coding": {
    "base_url": "https://.../v1",
    "api_key": "sk-..."
  },
  "eval": {
    "base_url": "https://.../v1",
    "api_key": "sk-..."
  },
  "models": {
    "debug": [
      "gpt-5.6-luna",
      "gpt-5.6-terra",
      "gemini-3.6-flash"
    ],
    "mature": [
      "claude-opus-5",
      "gpt-5.5",
      "gpt-5.6-sol",
      "gemini-3.1-pro-preview-high",
      "GLM-5.2",
      "Kimi-K3",
      "qwen3.7-max"
    ]
  }
}
```

The user fills in the real `base_url` / `api_key` values; agents only read the file. `models` lists the exact model names the relay expects, by tier. The eval harness must load `eval` credentials and model names from this file.

### Relay protocol support (gpt.ge, verified)

`base_url` is the host only (`https://api.gpt.ge`); clients append the protocol path:

| Protocol | Path | Works for | Use with |
|----------|------|-----------|----------|
| OpenAI chat completions | `/v1/chat/completions` | all models in `models` (GPT, Gemini, Claude, GLM, Kimi, Qwen) | Codex `wire_api = "chat"`, OpenAI-compatible SDKs, eval harness |
| OpenAI Responses | `/v1/responses` | OpenAI-family models (GPT); Claude returns 501 not implemented | Codex `wire_api = "responses"` |
| Anthropic Messages | `/v1/messages` | Claude models (native protocol) | Claude Code / Anthropic SDK, not Codex |

For Codex agent operations: use `wire_api = "chat"` (covers every model above, including Claude) or `wire_api = "responses"` (OpenAI-family models only). The Anthropic `/v1/messages` path is for Claude Code / Anthropic SDK clients, not Codex.

### Rules for agents

1. Read keys, base URLs, and model names from `~/.agents/relay.json`; never hard-code credentials or model names in code, prompts, or specs.
2. Use `eval` for all benchmark-evaluation calls; use `coding` only for the coding agent's own tooling. Keep the two separate.
3. If the file is missing, a key is empty, or a tier/model is missing, stop and ask the user — do not guess or substitute another key/provider.
4. Never print, echo, or write any `api_key` into outputs, logs, prompts, question specs, or git-tracked files.
5. **No token caps:** never set `max_tokens` or any token limit on model calls; models must generate without a cap.
6. **Reasoning effort:** default to `high` for all evaluation calls.

### Evaluation protocol (ArchitectureIQ)

1. **Tiers:** `debug` models (gpt-5.6-luna, gpt-5.6-terra, gemini-3.6-flash) for fast daily iteration; `mature` models (claude-opus-5, gpt-5.5, gpt-5.6-sol, gemini-3.1-pro-preview-high, GLM-5.2, Kimi-K3, qwen3.7-max) for mature-setting runs.
2. **First run:** once the harness is set up, start with **claude-opus-5 on 20–50 questions** to estimate difficulty. Inspect cases to confirm the evaluation is fair and fully exercises model capability before scaling to more models or larger batches.
3. **Scale gate:** multi-model runs with **total requests ≥ 300 questions** require mature settings and **explicit user approval** before launching.
4. **Final leaderboard:** before locking the leaderboard, ask the user to confirm the reasoning effort per provider; default to the **highest reasoning-effort value each provider supports**.
