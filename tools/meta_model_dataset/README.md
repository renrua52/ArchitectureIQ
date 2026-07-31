# Setting-to-loss meta-model dataset

This experiment tests whether a small meta-model can predict final loss from a
training setting.  The first phase is deliberately **in-distribution**: each
model sees one fixed dataset, budget, batch size, and loss, while model and
optimizer settings vary.

The unit of data is one independently sampled and executed **candidate
setting**, not one multiple-choice question.  Generating 1,000
significance-filtered questions would select examples using their labels and
would reuse candidates across questions, inflating the apparent sample size.

## Phase A environments

`plan_60q_id_v1.json` exactly matches the three environments behind the old
60-question evaluation:

| experiment | dataset | budget | batch | steps | fixed loss | varying axes |
|---|---|---:|---:|---:|---|---|
| univariate | `sym_62678b` | 2,048 | 32 | 64 | MSE | model, optimizer |
| multivariate | `mvar_c59a30` | 5,120 | 32 | 160 | MSE | model, optimizer |
| bigram LM | `bg_0021c1` | 5,120 | 64 | 80 | cross-entropy | model, optimizer |

Each experiment samples 1,000 primary settings plus 50 pre-declared reserves.
The split is assigned **before GT** and stratified by optimizer type and model
width.  It contains 900 train rows and 100 locked validation rows.  Treat the
100 as a final holdout: tune or cross-validate only within the 900 rows.

The original 64-candidate set for each environment is excluded by full spec
fingerprint before sampling.  This preserves the old 60 questions as a
secondary external evaluation rather than leaking their candidates into the
new train/validation data.

## Ground-truth and parameter-count invariants

Every target follows the repository's canonical path:

```text
candidate spec
  -> write_candidate()
  -> generated model.py/loss.py/optimizer.py/train.py
  -> run_ground_truth()
  -> results/summary.json
  -> summary[f"mean_{selection_metric}"]
```

The builder uses 10 shared initialization seeds, matching profile `v1`.  The
plan's `finite_only` mode overrides the benchmark's quality cutoff while running
GT, so finite but very bad settings retain a real regression target.  It does
not replace the training loop.  Each row also records
`target.benchmark_eligible`, which indicates whether all seed losses satisfy
the original dataset cutoff; report results both on all random settings and on
this question-eligible subset.

`derived.total_params` is not a hand-maintained formula.  The builder imports
the `Model` used by generated `train.py`, instantiates it, and sums
`model.parameters()`.  It then requires the registered model plugin's
`build_module()` count to match.  The row stores:

- `derived.total_params` — the requested explicit input variable;
- `derived.trainable_params`;
- `derived.log_total_params` — usually the better-scaled linear/MLP input.

Full SHA-256 spec fingerprints are used for uniqueness and splitting.  The
normal six-hex `candidate_id` is retained only as a display/provenance field;
at 1k scale it is too short to be a safe identity.

## Build

From the repository root:

```bash
# Fast: sample specs, freeze split, and render candidate files.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_60q_id_v1.json \
  --stage prepare

# Optional resumable timing smoke: ten pending candidates per experiment.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_60q_id_v1.json \
  --stage gt --workers 8 --limit-per-experiment 10

# Resume and finish all GT, then export. Completed candidates are skipped.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_60q_id_v1.json \
  --stage all --workers 8
```

Use `--experiment ID` to run one environment.  Including the pre-declared
reserves, the full three-environment run executes 31,500 seed trainings (about
3.192 million optimizer steps plus test
evaluation at every step); on this repository's previous 8–9 worker runs,
allow roughly 1–2 hours.  Output is gitignored under:

```text
data/meta_model/setting_to_loss_60q_id_v1/
  run_manifest.json
  {experiment_id}/
    sampling_manifest.json  # full specs and split, frozen before GT
    candidates/...          # generated code + summary.json + curves.npz
    attempts.jsonl           # primary and reserve attempts, including failures
    all.jsonl                # exactly 1,000 selected usable rows
    train.jsonl              # exactly 900
    validation.jsonl         # exactly 100 locked holdout rows
    feature_schema.json       # fitted from train covariates only
    manifest.json
```

Rerunning with unchanged inputs is idempotent.  Resume markers bind the GT
configuration to hashes of the candidate spec, all four generated Python files,
materialized dataset files, ArchitectureIQ execution sources, Python/PyTorch,
NumPy, and the stored summary/curves files.  A code, data, environment, or result
change therefore invalidates the marker and reruns that candidate instead of
mixing executions.
Changing a sampling/split setting causes a config-hash error; use a new
`output_root` for a new design.

## Row contract

Each selected JSONL row contains:

- provenance: experiment, profile, family/dataset, full fingerprint, paths and
  file hashes;
- raw `setting`: model, optimizer, loss, and budget;
- `derived`: exact parameter counts;
- flat `features`: setting-only model inputs;
- `target`: metric, mean/log-mean/std loss, seed/failure audit fields, and
  benchmark eligibility.

IDs, paths, seeds, standard deviation, failure flags, and all target values are
excluded from `features`.  `feature_schema.json` is fitted from the 900 train
rows only and describes numeric, categorical, unknown-category, and
missing-value handling without inspecting validation covariates or labels.

## Evaluation after the data is built

Train one model per experiment.  Recommended comparisons are:

1. constant mean;
2. parameter-count-only linear regression;
3. full linear/ridge regression;
4. small MLP;
5. decision forest or XGBoost (optional dependency).

Report raw/log MAE, RMSE, R², and rank correlation.  For ArchitectureIQ-style
selection, form choices only from holdout rows and report argmin accuracy plus
regret.  If a probability of being best is required, calibrate
`softmax(-predicted_loss / temperature)` using only folds within the 900-row
training set, then report holdout NLL/Brier score.

The earlier `fit_candidate_final_metric_rich` result (49/60, 81.7%) is useful
motivation but not a held-out result: its 193-row fitting table contains all 115
unique candidates used by those 60 questions.  This split and external-set
exclusion are designed to remove that leakage.

## Phase B: wide-distribution v2

`plan_wide_v2.json` is the scale-up plan.  It uses the separate
`meta_wide_v2` profile so neither v1 nor v2 is mutated after earlier artifacts
were frozen.  The prepared design contains:

- 10,000 primary settings plus 510 pre-declared reserves;
- 9,000 train and 1,000 locked validation labels, assigned before GT;
- 30 environments over 15 dataset instances (five per family);
- five budgets and five batch sizes;
- all compatible losses, five optimizer types, learning rates from `1e-5` to
  `1e-1`, wider regularization, MLP widths `8..384`/depths `1..8`, and
  transformer widths `16..192`/depths `1..5`;
- `b1_pilot` (3,006 primary settings) and `b2_scale` (6,994), selectable with
  `--phase`.

Every sampling record freezes top-level `group_labels` before GT: `phase`,
`family`, `dataset`, `environment`, and `dataset_cohort`.  These are not copied
into `features`.  Use `environment` or `dataset` for grouped CV and the
`holdout_candidate` cohort for leave-one-dataset-out analysis.  The ordinary
`split` remains the exact per-environment 90/10 locked split; the cohort label
does not silently remove rows from it.

The three old 60-question candidate sets and all 3,150 Phase-A attempts
(including reserves) are exclusion sources.  They are filtered by full spec
SHA-256 for the relevant dataset, before sampling.  The frozen config hash
includes the source hashes, relevant exclusion fingerprints, profile, split,
phase, and group labels.  GT resume additionally binds generated code, dataset
files, execution sources/environment, and result hashes as described above.

Prepare and audit without starting GT:

```bash
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json --stage prepare

.venv/bin/python -m tools.meta_model_dataset.audit_prepare \
  --plan tools/meta_model_dataset/plan_wide_v2.json
```

The 2026-07-14 prepare audit is recorded in
`wide_v2_prepare_audit.md`; the complete machine-readable report is generated
under `data/meta_model/setting_to_loss_wide_v2/prepare_audit.json`.  It verifies
10,510 globally unique attempts, zero excluded overlap, all rendered candidate
inputs present, exact split totals, and zero GT markers/summaries at prepare
time.

Run B1 first, then B2 only after checking runtime and unusable/replacement rates:

```bash
# Optional small timing/finite-loss smoke: 3 candidates in each B1 environment.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json \
  --stage gt --phase b1_pilot --limit-per-experiment 3 --workers 8

# Resumes the smoke, finishes B1, and exports B1 environments.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json \
  --stage all --phase b1_pilot --workers 8

# Scale only after the B1 audit passes.
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json \
  --stage all --phase b2_scale --workers 8
```
