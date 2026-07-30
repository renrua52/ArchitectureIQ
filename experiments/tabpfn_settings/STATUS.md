# Status

## Done (generated per-family corpus)

Pipeline `scripts/run_family_pipeline.py` on A100 (`v2.5-tabpfn-settings`, count=40, vary=model+optimizer, n_seeds=3):

| Family | dataset_id | target | N finite | TabPFN Spearman | TabPFN pairwise | Ridge pairwise |
|--------|------------|--------|----------|-----------------|-----------------|----------------|
| univariate_regression | sym_f91a02 | mean_test_mse | 36 | 0.35 | 0.62 | 0.66 |
| multivariate_regression | mvar_aa9fd3 | mean_test_mse | 39 | 0.51 | 0.68 | 0.73 |
| bigram_lm | bg_9ef717 | mean_test_ce | 40 | 0.70 | 0.75 | 0.69 |
| synthetic_tabular_classification | stabcls_c4d0e7 | mean_test_ce | 40 | 0.71 | 0.76 | 0.74 |

HTML: `artifacts/generated/report_families_generated.html`

## Protocol

- **X:** candidate_spec features (model/opt/loss/budget)
- **y:** GT `mean_*` from executing generated `train.py`
- **TabPFN train size:** per-fold n_train (~28–32); no weight fine-tune

## Next

Scale count (e.g. 200+) and/or vary loss; optional larger n_seeds.
