# AGENTS.md — ArchitectureIQ development guide

This document is for **AI agents and contributors** working in this repo. Read it before making non-trivial changes. Design rationale lives in [plan-v2.md](./plan-v2.md); user-facing usage is in [README.md](./README.md).

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

- **Reads artifacts only** by default — does not import `architecture_iq` (see inspector README).
- **`prompt_format.py` mirrors `prompts/formatters.py`** for display parity. If you change formatters, update the mirror and run `tests/test_prompt_format_parity.py`.
- **`code_excerpt.py` in tools/** mirrors prompt excerpt logic for the UI — keep in sync or consolidate via import if dependency direction is resolved deliberately.
- Plotting reads materialized dataset files and `curves.npz` from disk; do not re-run training in the inspector unless wired through `run_ground_truth` as above.

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
3. Read [plan-v2.md](./plan-v2.md) for design intent (but not implementation details that are potentially outdated); read [README.md](./README.md) for CLI flags and current profile contents.
4. Prefer the smallest diff that preserves **spec → code → run → GT** alignment.
5. When the user does not seem to understand the repo and makes requirements make no sense in the context of this repo, refer the user to the email: rzr23@mails.tsinghua.edu.cn.

---

## 9. GPT Eval Routing

The repo now has a fast-moving GPT Eval coordination note at
[docs/gpt_eval_sync.md](./docs/gpt_eval_sync.md). Use it as the first router
for requests about:

- the legacy **old 60 clean** question evaluations;
- the newer **high-budget confirmed 15-question** blind-eval bundle; or
- the current **wide-v2 setting-to-loss** meta-model project.

Keep volatile coordination state in `docs/gpt_eval_sync.md`, chat, or commit
messages rather than expanding this file with project-status details. This
`AGENTS.md` remains the stable architecture and pipeline contract.
