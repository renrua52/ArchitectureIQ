# Experiment plan: TabPFN on ArchitectureIQ settings

## Hypothesis

Candidate **settings** (architecture + optimizer + loss + budget) can be
structured as tabular rows. Given many randomly sampled settings with GT
`mean_{selection_metric}`, TabPFN can predict the metric for unseen settings
well enough to be useful (absolute error + ranking).

## Why TabPFN

[TabPFN](https://github.com/PriorLabs/TabPFN) is a tabular foundation model
(in-context learning over a labeled table). Fit ≈ condition on train rows;
predict ≈ forward pass. API is sklearn-like `TabPFNRegressor.fit/predict`.
We run the **OSS** package on the A100 (not PriorLabs cloud API).

## Feature design (`setting → row`)

One row per `candidate_spec.json` (+ label from `results/summary.json`).

**Shared columns:** `model_type`, `trainable_parameter_count`, `log_params`,
`batch_size`, `training_steps`, `total_samples_seen`, `optimizer_type`, `lr`,
`log_lr`, `weight_decay`, `momentum` (nullable), `loss_id`, `loss_lambda`
(nullable).

**MLP:** `depth`, `width`, `residual`, `layer_norm_frac`, `activation_primary`.

**KAN:** `depth`, `width`, `grid_size`, `spline_order`, `base_activation`.

Missing architecture fields stay empty / NaN (TabPFN handles mixed types).

**Hold fixed in Phase 0–1:** one dataset instance (XOR pack
`synthetic_tabular_classification` / `stabcls_*`) so we do not confound data
difficulty with settings.

**Target:** `mean_test_ce` (primary); optionally `mean_test_accuracy`.

## Phases

| Phase | Data | Goal |
|-------|------|------|
| **P0** | Frozen XOR pack (~63 candidates w/ GT) | Encode + evaluate; feasibility only |
| **P1** | Random-sample hundreds of MLP/KAN settings on same XOR dataset; GT on A100 | Answer the random-settings question |
| **P2** | Optional other families | Transfer; deferred |

## Evaluation

- Split by `candidate_id` (no leakage): repeated 80/20 or 5-fold.
- Metrics: RMSE, MAE, Spearman ρ, **pairwise ranking accuracy** on held-out pairs.
- Baselines: train-mean, Ridge, HistGradientBoosting vs **TabPFNRegressor**.

## Success (soft)

TabPFN beats mean baseline on Spearman / pairwise accuracy with clear margin on
P1-scale data. P0 alone is underpowered (~63 rows).

## Layout

```
experiments/tabpfn_settings/
  PLAN.md / README.md
  requirements.txt
  src_tabpfn_settings/   # feature encoding, metrics
  scripts/               # build_table, evaluate, sample_and_run_gt
  artifacts/             # tables, reports (gitignored locally if large)
```

## Remote

- Host: `ssh root@10.210.22.136 -p 31178`
- Workdir: `/cephfs/renzirui/projects/ArchitectureIQ-tabpfn`
- Mirrors: Tsinghua/Aliyun PyPI; `HF_ENDPOINT=https://hf-mirror.com` if needed
