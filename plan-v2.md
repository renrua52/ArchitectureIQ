# ArchitectureIQ Benchmark Plan

This document defines the **general** ArchitectureIQ benchmark architecture — pools, schemas, invariants, and pipeline — so new dataset families, model types, and question types can be added without redesign. [plan.md](./plan.md) is the original sketch; this file makes the design explicit and extensible. **Terminology:** [AGENTS.md](./AGENTS.md#terminology).

**V1** is not a different benchmark — it is the **first implementation profile**: one dataset family, one model type, and a small set of optimizers/losses. Everything below is written as if the full system already exists; [V1 Profile](#v1-profile-first-implementation) at the end records what V1 actually ships.

---

# Motivation

Architecture IQ benchmark 回答一个基础的问题：
LLM 是否已经具备对于最简单模型、最简单数据集的 Architecture Intuition 和 Learning Mechanics 理解能力？

ArchitectureIQ deliberately starts with the simplest, fully synthetic, fully controllable settings — not complex LLMs or real corpora — so that:

1. Real-world dataset complexity does not confound what we measure about model understanding.
2. All experiments are programmatically generatable and arbitrarily scalable.
3. Every variable is controlled, making analysis of what was understood tractable.
4. The same framework extends later to harder dataset families, model families, and training regimes.

ArchitectureIQ is an **Architecture Intelligence IQ Test** framework, not a monolithic fixed benchmark.

---

# Design Principles

## Extensibility first

The codebase and on-disk layout should treat **pools** as first-class registries:

| Pool | Key | Scoped by |
|------|-----|-----------|
| Dataset | `family` → many `dataset_id` instances | — |
| Model | `model.type` (model type) | compatible with dataset family |
| Optimizer | `optimizer.type` | global (any candidate) |
| Loss | `loss_id` | **dataset family** (each family declares its loss pool) |
| Training budget | `total_samples_seen` | global predefined grid |
| Question type | `type` tag | any compatible pool combination |

Adding a new dataset family means: register the family, its synthesis code, its selection metric, its loss pool, and (optionally) family-specific significance thresholds. Existing families and questions remain valid.

## Single source of truth

For every artifact, **spec JSON + checked-in code** must match what the ground-truth runner executes. Prompts are rendered from specs via templates — never hand-written in parallel.

## Profile-based rollout

A **profile** (e.g. `v1`) selects subsets from each pool. Profiles are declared in config (`profiles/v1.yaml`), not hard-coded in pipeline logic. The pipeline always speaks the general language of pools; profiles only constrain sampling.

---

# Core Invariants (Fair Comparison Contract)

Every valid multiple-choice question MUST satisfy all of the following, regardless of profile or family:

| Invariant | Rule |
|-----------|------|
| Same dataset instance | Identical train and test tensors (format defined by dataset family) for all choices |
| Stated budget per choice | Each choice's `total_samples_seen = training_steps × batch_size` is explicit in its spec and prompt |
| Shared budget within one set | All candidates in a single generated set share the same `total_samples_seen` (set at `--budget`) |
| Cross-budget across sets | `generate-question` may union candidates from multiple sets with different budgets; `question.json` marks `budget.mixed` and prompts state per-choice budgets |
| Same selection metric | Rank choices by the metric declared in `dataset_spec.json` (`selection_metric`) |
| One correct answer | Exactly one choice is labeled correct; ties or ambiguous rankings reject the question |
| No metric leakage | Prompts describe setups only — no final metrics, curves, or seed statistics |
| Reproducibility | Ground truth records environment metadata (see [Reproducibility](#reproducibility)) |

Family-specific rules (domain, input dimensionality, metric name, fail thresholds) live in **family config**, not in question logic.

**Training budget definition:** `total_samples_seen = training_steps × batch_size`. One optimizer step consumes one mini-batch. Within a candidate set, `total_samples_seen` is fixed; across sets, questions may intentionally mix budgets.

**Budget grid (global):** Candidate generation picks from a predefined set of budgets in the profile — never a one-off random budget at sample time. Cross-budget questions combine candidates trained at different grid values.

---

# Pipeline Overview

The pipeline is family-agnostic; family plugins implement synthesis, training hooks, and metric computation.

```
1. Dataset synthesis     → dataset_spec.json + materialized data (family-specific)
2. Candidate generation  → candidate_spec.json + runnable code (from pools)
3. Ground-truth training → metrics per seed, aggregated stats
4. Question assembly     → select choices, validate significance, tag type
5. Prompt rendering      → family-aware templates → prompt (no GT metrics)
6. Evaluation            → compare test-taker letter to ground truth
```

Steps 4–6 operate on specs and results the same way for every family.

---

# Dataset Pool

The **dataset pool** is a registry of **dataset families**. Each family defines how data is synthesized, what tensors are stored, what metrics apply, and which loss functions are valid.

## Family interface (contract)

Every dataset family MUST implement:

| Responsibility | Artifact / field |
|----------------|------------------|
| Identity | `family` string (e.g. `univariate_regression`) |
| Instance ID | `dataset_id` unique within family |
| Synthesis | `synthesize.py` — reproducible from spec |
| Target sampler (optional) | Procedural generator for symbolic targets; sampled params frozen in spec |
| Materialized data | Family-defined files (e.g. `train.pt`, `test.pt`) |
| NL description | Template key `dataset/{family}.md` |
| Selection metric | `selection_metric` in spec (used to rank choices) |
| Training objective | Declared per candidate (loss pool); may differ from selection metric |
| Loss pool reference | `loss_pool_id` or inline list of valid `loss_id` values |
| Significance config | Optional overrides: `gap_min`, `fail_threshold`, `eval_interval_steps` |
| Compatibility | Which `model.type` values are allowed |

## Registered families

| `family` | Status | Description |
|----------|--------|-------------|
| `univariate_regression` | **V1** | Scalar input/output; symbolic target on a bounded domain via expression sampler |
| `multivariate_regression` | planned | Vector input, scalar or vector output |
| `classification` | planned | Multi-class classification |

New rows extend the pool; schemas and folder layout stay stable.

## Dataset instances

Within a family, each **dataset instance** is one concrete draw (e.g. one sampled symbolic target + point-sampling seed + split sizes):

- Described by `dataset_spec.json` (family-specific fields in a nested `params` object).
- Materialized once; all candidates for that instance reference the same tensors.
- Many instances per family; variants come from the family’s **target sampler** (or fixed catalog), domain, noise, etc.

### Symbolic target sampler (`univariate_regression`)

Families that regress to closed-form targets SHOULD support a **symbolic expression sampler** rather than a fixed function catalog:

1. Sample an expression tree from a grammar (ops, depth, coefficients) using a spec seed.
2. **Validate** the candidate (non-constant, sufficient curvature, bounded range on domain — family-defined rules).
3. **Freeze** the result: canonical expression string + full `params` written to `dataset_spec.json`.
4. Emit `synthesize.py` that evaluates exactly that expression (the prompt shows this code).

Re-sampling the same seed MUST reproduce the same instance. The sampler is a family plugin; profiles constrain its grammar (see [V1 Profile](#v1-profile-first-implementation)).

## Artifacts (per instance)

```
data/datasets/{family}/{dataset_id}/
  dataset_spec.json
  synthesize.py
  <materialized files per family>
  candidates/
    budget_{total_samples_seen}/
      ...
```

## Description for test takers

For every family:

- Natural language via **string templates** keyed by `family`.
- Synthesis code included in the prompt (full source in prompt for now; family template may shorten later).

---

# Model Pool

The **model pool** is a registry of **model types** (`model.type`). Each type defines a parameter schema, code generator, and NL template.

## Model family interface

| Responsibility | Detail |
|----------------|--------|
| Identity | `model.type` (e.g. `mlp`, `cnn`) |
| Schema | JSON object in `candidate_spec.json` under `model` |
| Code | `model.py` implementing `torch.nn.Module` |
| NL template | `model/{type}.md` |
| Compatibility | Subset of dataset families that may use this type |

## Registered model types

| `model.type` | Status | Notes |
|--------------|--------|-------|
| `mlp` | **V1** | Fully-connected stacks; optional residual, layer norm, per-layer activations |
| `cnn` | planned | conv nets for image datasets |

## Architecture dimensions (family-dependent)

Each model type declares which dimensions exist and how they are sampled. Examples across types:

- **MLP:** depth, width, residual, layer norm per layer, activations, init scheme.
- **CNN (planned):** channel widths, kernel sizes, depth, pooling, activations — for image `classification` family.
- **Init (future):** per-type pool (`default`, `xavier`, `kaiming`, …).

Invalid combinations are rejected at **generation time** by the type’s validator, not at training time.

---

# Optimizer Pool

The **optimizer pool** is global — any candidate may use any registered optimizer.

## Optimizer interface

| Field | Detail |
|-------|--------|
| `optimizer.type` | e.g. `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad` |
| Hyperparameters | Type-specific schema in `candidate_spec.json` |
| Code | `optimizer.py` or shared factory keyed by type |
| NL template | `optimizer/{type}.md` |

Hyperparameters are drawn from **profile-defined grids** and frozen in the spec (not re-sampled at run time).

## Registered optimizers

| `optimizer.type` | Status |
|------------------|--------|
| `SGD` | **V1** |
| `Adam` | **V1** |
| `AdamW` | **V1** |
| `RMSprop` | **V1** |
| `Adagrad` | **V1** |

---

# Loss Function Pool

The **loss pool is keyed by dataset family** — not global. Each dataset family declares which losses are meaningful for its task.

## Loss interface

| Field | Detail |
|-------|--------|
| `loss_id` | Unique within a family (e.g. `mse`, `bce`, `mse_l2`) |
| Definition | Documented + implemented in `loss.py` |
| NL template | `loss/{family}/{loss_id}.md` |
| Extra params | e.g. regularization `lambda`, class weights |

Training objective (loss) and selection metric (from dataset spec) **may differ**; both must appear in the prompt and spec.

## Loss pools by family

### `univariate_regression` (V1)

| `loss_id` | Definition |
|-----------|------------|
| `mse` | Mean squared error |
| `mse_l2` | MSE + λ · mean(w²) |
| `mse_l1` | MSE + λ · mean(\|w\|) |

### `classification` (planned)

| `loss_id` | Definition |
|-----------|------------|
| `ce` | Cross-entropy |
| `ce_l2` | Cross-entropy + L2 weight penalty |

---

# Candidate Pool

A **candidate** is a complete training setup: `(dataset_id, model, optimizer, loss, budget, hyperparameters)`.

## Artifacts (per candidate)

```
.../candidates/budget_{total_samples_seen}/{candidate_id}/
  candidate_spec.json
  model.py
  train.py
  loss.py
  optimizer.py
  results/
    summary.json
    curves.npz
    seeds/{seed}/   # optional
```

`dataset_spec.json` + `candidate_spec.json` + pinned code MUST fully determine behavior.

## Pool organization

- Candidates for the **same dataset instance** live under that instance’s folder.
- Candidates for the **same training budget** live under the same `budget_*` subfolder.
- Cross-family candidates never share a folder — family is implicit in the path.

---

# Ground Truth

## Training runs

For each candidate:

1. Run `n_seeds` independent trainings (count from profile or family config).
2. Each run records:
   - Final **selection metric** (from `dataset_spec.selection_metric`).
   - Progressive metric trace at `eval_interval_steps` (family config), aligned to optimizer steps.
3. Aggregate across seeds: mean, std, optional per-step curves.

Analysis artifacts may include curves; **prompts never include them**.

## Failure handling

| Outcome | Action |
|---------|--------|
| NaN/Inf | Mark run failed |
| Metric worse than `fail_threshold` | Mark run failed (threshold from family config) |
| Too many failed seeds | Exclude candidate from question pool (threshold from profile) |

## Significance (question validation)

Significance rules are **metric-aware** and use family defaults unless overridden in `dataset_spec.json`:

1. Let `μᵢ` = mean selection metric for choice `i` (lower is better unless family declares `higher_is_better`).
2. Sort choices; let `C*` be best, `μ*` its mean, `μ₂` runner-up mean.
3. **Gap:** `μ₂ − μ* ≥ gap_min` (family or profile default).
4. **Stability:** Winner beats all others in ≥ `win_rate_min` fraction of seeds.
5. **Non-overlap (optional):** `μ* + std* < μ₂ − std₂`.

Failed validation → discard question (targeted mode: regenerate; retrospective mode: resample).

Stored on `question.json`:

```json
"significance": { "passed": true, "gap": 0.12, "win_rate": 0.9, "metric": "test_mse" }
```

---

# Question Generation

## Question types (tags)

Types describe **what is held constant vs varied** across choices. They apply to any dataset family / model type the pools support.

| Tag | Typically held constant | Typically varied |
|-----|-------------------------|------------------|
| `architecture_only` | dataset, budget, optimizer, loss, training hparams | `model.*` only |
| `optimizer_only` | dataset, budget, model, loss | `optimizer.*` only |
| `loss_only` | dataset, budget, model, optimizer | `loss.*` only |
| `data_only` | budget, model, optimizer, loss | dataset instance within same family (future) |
| `mixed` | dataset, budget | any combination |

Every question records:

- `type` — tag from above
- `generation_mode` — `targeted` or `retrospective`
- `family` — dataset family (for filtering and reporting)

## Generation modes

### `targeted`

1. Input: question template `{ type, family, dataset_id?, budget, num_choices, constraints? }`.
2. Generate candidates satisfying constraints (or reuse matching pool entries).
3. Run ground truth for new candidates.
4. Validate significance; retry up to `max_attempts`.

Best when the pool lacks structured contrasts (often `architecture_only` on a new family).

### `retrospective`

1. No new candidates.
2. Search existing results under `{family, dataset_id, budget}`.
3. Sample `num_choices` matching `type`.
4. Validate significance; resample if invalid.

Best for recycling strong comparisons (`mixed`, large pools).

**Mode and type are orthogonal** — either mode can serve any type if the pool supports it.

## Question format

- `num_choices` from profile (default 4 → letters A–D; extensible to N).
- One correct letter = best mean selection metric.
- Choice order shuffled at render time; mapping stored in `question.json`.

---

# Prompt Generation

## Structure (family-aware)

1. Global task instructions (template `prompt/header.md`).
2. Dataset section — template `dataset/{family}.md` + `synthesize.py`.
3. Budget and evaluation metric (from dataset spec).
4. Per choice — model, optimizer, loss sections from type/family templates + code files.

## Excluded

- Final or progressive metrics, seed stats, significance, correct answer.

## Response format

Single letter `A`–`Z` depending on `num_choices`; case-insensitive exact match.

## Natural language strategy

- **Default: string templates only** — no LLM API required.
- Templates live in `prompts/templates/{family,type,...}.md`.
- Optional **LLM paraphrase** stage (future) must be versioned, logged, and off by default.

---

# JSON Schemas

Schemas use a stable envelope; family- and type-specific fields sit in nested objects so new pools do not break old specs.

## `dataset_spec.json`

```json
{
  "schema_version": "1.0",
  "dataset_id": "sym_a1b2c3",
  "family": "univariate_regression",
  "params": {
    "sampler": {
      "id": "symbolic_expr",
      "seed": 42,
      "max_depth": 3
    },
    "expression": "sin(6.283*x) * (x**2 + 0.3) + 0.2 * cos(4.0*x)",
    "domain": [0.0, 1.0],
    "train_size": 256,
    "test_size": 256,
    "noise": { "enabled": false },
    "point_sampling": { "distribution": "uniform", "seed": 1042 }
  },
  "selection_metric": "test_mse",
  "significance": {
    "gap_min": 0.05,
    "fail_threshold": 2.0,
    "eval_interval_steps": 50
  },
  "files": {
    "synthesize": "synthesize.py",
    "train": "train.pt",
    "test": "test.pt"
  }
}
```

Other families reuse the envelope; `params` shape differs (documented per family).

## `candidate_spec.json`

```json
{
  "schema_version": "1.0",
  "candidate_id": "c_7f3e91",
  "dataset_id": "sym_a1b2c3",
  "family": "univariate_regression",
  "budget": {
    "training_steps": 312,
    "batch_size": 32,
    "total_samples_seen": 9984
  },
  "model": {
    "type": "mlp",
    "depth": 3,
    "width": 64,
    "residual": true,
    "layer_norm": [true, false, true],
    "activations": ["gelu", "relu", "silu"]
  },
  "optimizer": {
    "type": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "betas": [0.9, 0.999]
  },
  "loss": {
    "loss_id": "mse_l2",
    "lambda": 0.001
  },
  "files": {
    "model": "model.py",
    "train": "train.py",
    "loss": "loss.py",
    "optimizer": "optimizer.py"
  }
}
```

Future model types replace the `model` object shape; `model.type` discriminates.

## `question.json`

```json
{
  "schema_version": "1.0",
  "question_id": "q_0042",
  "profile": "v1",
  "family": "univariate_regression",
  "dataset_id": "sym_a1b2c3",
  "budget": { "total_samples_seen": 10000, "training_steps": 625, "batch_size": 16 },
  "type": "architecture_only",
  "generation_mode": "targeted",
  "num_choices": 4,
  "choices": [
    { "letter": "A", "candidate_id": "c_7f3e91", "candidate_path": "..." },
    { "letter": "B", "candidate_id": "c_8a1b22", "candidate_path": "..." },
    { "letter": "C", "candidate_id": "c_9c3d44", "candidate_path": "..." },
    { "letter": "D", "candidate_id": "c_0e5f66", "candidate_path": "..." }
  ],
  "correct_letter": "B",
  "significance": {
    "passed": true,
    "gap": 0.12,
    "win_rate": 0.9,
    "metric": "test_mse"
  },
  "prompt": {
    "template_version": "1.0",
    "rendered_path": "prompt.txt"
  }
}
```

---

# File Structure

```
data/
  datasets/
    {family}/
      {dataset_id}/
        dataset_spec.json
        synthesize.py
        ...
        candidates/
          budget_{total_samples_seen}/
            {candidate_id}/
              ...

  questions/
    {question_id}/
      question.json
      prompt.txt

profiles/
  v1.yaml              # subset of each pool for first release
  v2.yaml              # future

prompts/
  templates/
    header.md
    dataset/{family}.md
    model/{type}.md
    optimizer/{type}.md
    loss/{family}/{loss_id}.md

benchmark_manifest.json
```

**ID conventions:**

- `dataset_id` = `{short_prefix}_{hash(params)}` (e.g. `sym_` for sampled symbolic targets)
- `candidate_id` = `c_{hash(full spec)}`
- `question_id` = `q_{sequential or hash}`

---

# Reproducibility

Record in `results/summary.json` and `benchmark_manifest.json`:

- Python, PyTorch, CUDA versions; device type
- `n_seeds`, `base_seed`, `profile`, `family`
- Git commit hash of benchmark code

Re-runs after code changes invalidate affected candidates; bump `schema_version` on breaking spec changes.

---

# Benchmark Evaluation (Meta)

| Aspect | Rule |
|--------|------|
| Scoring | Exact match on `correct_letter` |
| Human baseline | Same `prompt.txt` as LLM |
| Reporting | Accuracy overall, by `family`, by `type`, by `profile` |
| Difficulty (optional) | Profile-defined bins from `significance.gap` |

---

# Implementation Order (Suggested)

Build the **general pipeline and plugin interfaces first**, then fill in the V1 profile.

1. Pool registries + family/type plugin interfaces (empty stubs for planned entries).
2. `univariate_regression` family + symbolic sampler + materialization + spec schema.
3. `mlp` model generator + validator.
4. Optimizer factory + regression loss pool.
5. Single-seed then multi-seed ground-truth runner.
6. Significance validator (reads family config).
7. Targeted question generator + `architecture_only`.
8. Prompt template renderer (family-aware).
9. Retrospective sampling + `mixed` questions.
10. Evaluation harness + `profiles/v1.yaml`.

---

# V1 Profile (First Implementation)

Profile `v1` selects the first row from each pool. **Everything above still applies**; this section only narrows what V1 samples and ships.

## Enabled pools

| Pool | V1 subset |
|------|-----------|
| Dataset families | `univariate_regression` only |
| Model types | `mlp` only |
| Optimizers | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad` |
| Losses | `mse`, `mse_l2`, `mse_l1` (regression family) |
| Question types | `architecture_only`, `optimizer_only`, `loss_only`, `mixed` |
| `num_choices` | 4 |

## `univariate_regression` instances (V1)

- Domain `[0, 1]`; i.i.d. uniform point samples; noiseless labels.
- Train 256 / test 256 points per instance.
- Targets come from the **`symbolic_expr` sampler** (not a fixed function list).

### Symbolic sampler (`symbolic_expr`, V1)

Sample a random expression tree, then reject/resample until validation passes.

**Grammar**

| Role | V1 choices |
|------|------------|
| Leaf | `x`, or rational constant in `[-2, 2]` (half-integers and thirds) |
| Unary | `sin(2π·)`, `cos(2π·)`, `tanh(2·)`, `abs(·)`, `(·)²`, `(·)³` |
| Binary | `+`, `-`, `*`, `/` (denominator clamped: `max(\|denom\|, 0.1)`) |
| Max depth | `3` (root at depth 0 → up to ~3 nested nonlinear ops) |
| Subtree reuse | Disabled in V1 (no shared subtrees) |

**Typical drawn forms** (illustrative, not exhaustive):  
`sin(2πx)·x² + 0.5x`, `tanh(cos(2πx)) / (0.3 + x)`, `x³ − sin(4πx)`, `|sin(2πx)| · (1 − x)`.

**Validation** (reject and resample with `seed + retry` if any fail):

| Rule | V1 threshold |
|------|----------------|
| Non-constant | Range on 128-point domain grid ≥ `0.4` |
| Non-trivial | At least one nonlinear op (sin/cos/tanh/abs/pow/div) |
| Bounded | `max \|y\|` on domain ≤ `5.0` |
| No near-singularity | ≥ 95% of grid points have `\|y\| ≤ 4.5` |

**Frozen spec fields:** `params.sampler`, `params.expression` (canonical infix string), `params.point_sampling.seed` separate from expression seed.

**Default significance (V1):** `fail_threshold = 2.0` on test MSE (tighter than trivial constants; expressions are harder than single-term baselines).

## `mlp` sampling (V1)

| Dimension | Values |
|-----------|--------|
| Topology | 1 → [hidden]×depth → 1; no I/O normalization |
| Depth | {1, 2, 3, 4} |
| Width | {16, 32, 64, 128} |
| Residual | {false, true} — skip around hidden blocks when dims match |
| LayerNorm | Per layer {false, true}; **pre-norm** when enabled |
| Activation | Per layer: {relu, leaky_relu, gelu, silu}; leaky slope 0.1 |
| Init | PyTorch `Linear` defaults |

## Optimizer grids (V1)

| Param | Values |
|-------|--------|
| Learning rate | {1e-4, 3e-4, 1e-3, 3e-3} |
| Weight decay | {0, 1e-5, 1e-4, 1e-3} |
| SGD momentum | {0, 0.9} |
| Adam/AdamW betas | (0.9, 0.999) |
| Batch size | {16, 32, 64} |

## Budget grid (V1)

`total_samples_seen ∈ {1_000, 5_000, 20_000}` — `training_steps = budget / batch_size`.

## Ground truth (V1)

- `n_seeds = 10`, `base_seed` from profile.
- `eval_interval_steps = 50`.
- `fail_threshold = 2.0` (test MSE; matched to symbolic target scale on `[0, 1]`).
- Exclude candidate if ≥ 3 seeds fail.
- Significance: `gap_min = 0.05`, `win_rate_min = 0.7`, non-overlap heuristic enabled.

## Prompts (V1)

- Templates only; no LLM paraphrase.
- Full `synthesize.py` and per-choice code in prompt.

## V1 non-goals

- Other dataset/model families (stubs only in registry).
- Custom inits beyond PyTorch defaults.
- Training until convergence.
- Metric leakage in prompts.
- LLM-generated natural language.

---

# Changelog from plan.md

| Area | Change |
|------|--------|
| Framing | General multi-pool benchmark + V1 profile (not V1-only doc) |
| Dataset | Family registry + instance model; regression is one family |
| Model / optimizer / loss | Separate extensible pools; loss scoped by dataset family |
| Profiles | `profiles/v1.yaml` constrains sampling without forking pipeline |
| Schemas | Stable envelope + nested `params`; `family` on all artifacts |
| File layout | `{family}/` tier under datasets; template paths by family/type |
| Invariants | Metric and thresholds from family config, not hard-coded MSE |
| V1 targets | Fixed function catalog → `symbolic_expr` procedural sampler with validation |
