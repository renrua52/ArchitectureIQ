# Wide-v2 prepare audit (2026-07-14)

Status: **PASS**.  This audit was run after `--stage prepare`; no ground-truth
job was launched.

- Plan: `tools/meta_model_dataset/plan_wide_v2.json`
- Plan SHA-256: `8ba994e1ba2ac168fa6f193911bfdc02de089358cfeca1764983d74b68af9338`
- Output: `data/meta_model/setting_to_loss_wide_v2/`
- Machine-readable audit: `prepare_audit.json` in that output directory

## Frozen scale

| item | count |
|---|---:|
| Environments | 30 |
| Dataset instances | 15 |
| Primary settings | 10,000 |
| Train primary | 9,000 |
| Locked-validation primary | 1,000 |
| Reserve settings | 510 |
| Total prepared attempts | 10,510 |
| B1 pilot attempts / primary | 3,159 / 3,006 |
| B2 scale attempts / primary | 7,351 / 6,994 |
| GT summaries | 0 |
| GT resume markers | 0 |

Family attempt counts are 3,504 univariate, 3,503 multivariate, and 3,503
bigram.  Each family spans five dataset instances and ten budget/batch
environments.

## Coverage observed in frozen specs

| axis | observed values |
|---|---|
| Budget | 1,024; 2,048; 5,120; 10,240; 20,480 |
| Batch size | 8; 16; 32; 64; 128 |
| Optimizer | SGD; Adam; AdamW; RMSprop; Adagrad |
| Learning rate | `1e-5` through `1e-1` (9 log-spaced values) |
| Weight decay | 0 through `1e-1` (7 values) |
| Loss | base, L1, and L2 variants for every family |
| Loss lambda | `1e-6` through `1` (7 values) |
| MLP depth / width | 7 depths (`1..8`); 7 widths (`8..384`) |
| Transformer layers / width | 5 layers (`1..5`); 8 widths (`16..192`) |
| Transformer heads / FF width | 4 head counts; 6 FF widths |

The least frequent optimizer type still has 2,052 attempts; the least frequent
learning rate has 1,114.  Architecture buckets are similarly balanced by the
seeded uniform sampler.  Full counts are in the generated audit JSON/Markdown.

## Integrity checks

- All 10,510 full candidate SHA-256 fingerprints are globally unique.
- Overlap with the old 60-question candidates and Phase-A sampling manifests is
  zero.  For each anchor dataset, 64 old-60 candidates plus 1,050 Phase-A
  attempts (1,114 fingerprints) were loaded before sampling.
- Every record has its train/validation split, primary/reserve role, stratum,
  and group labels frozen in `sampling_manifest.json` before GT.
- All records have matching `candidate_spec.json`, `model.py`, `loss.py`,
  `optimizer.py`, and `train.py` rendered through `write_candidate()`.
- Every sampling-manifest config hash recomputes exactly.  Phase/group labels,
  profile contents, dataset spec, budget/batch, GT settings, and exclusion
  source/fingerprint hashes are part of that resume identity.
- The audit found zero result summaries and zero GT markers; it did not inspect
  any targets or curves.

## Gate before B2

Run the 3,006-setting B1 pilot first.  Before enabling B2, inspect wall time,
finite-loss rate by optimizer/LR/loss, reserve consumption per split/stratum,
and whether very large regularization/LR cells should remain as valid wide-tail
examples.  Do not change this output root after GT begins; a design change gets
a new plan/output root so resume hashes cannot mix distributions.
