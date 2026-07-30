# Status

## Done

- Branch: `side/tabpfn-settings`
- Packs evaluated (all tracked on branch):
  - XOR `xor-v2.5-100q-37b9da` → 63 rows (`stabcls_953608`)
  - GRU `gru-v2.5-100q-a48abc` → 48 rows (`bg_fdc03b`)
- TabPFN v2.5 regressor, 5-fold CV, target `mean_test_ce`, A100
- Detailed HTML: `artifacts/report_p0_detailed.html`
- JSON: `artifacts/report_xor.json`, `artifacts/report_gru.json`

### Headline OOF metrics

| Pack | Model | MAE | Spearman | Pairwise |
|------|-------|-----|----------|----------|
| XOR | TabPFN | 0.020 | 0.721 | 0.772 |
| XOR | Ridge | 0.031 | 0.567 | 0.692 |
| GRU | TabPFN | 0.022 | 0.860 | 0.868 |
| GRU | Ridge | 0.075 | 0.277 | 0.604 |

## Protocol (short)

- **Input X:** flattened candidate_spec features (model/opt/loss/budget); see report.
- **Output y:** GT `mean_test_ce` from `results/summary.json`.
- **Train size:** per-fold n_train (XOR 50–51, GRU 38–39). TabPFN does not fine-tune weights.

## Next

P1: random-sample settings on the same dataset instances + GT on A100.
