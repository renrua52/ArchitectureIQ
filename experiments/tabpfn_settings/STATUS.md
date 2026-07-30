# Status

## Approach (corrected)

Do **not** treat frozen question packs as the primary TabPFN corpus.
For each dataset **family**, run the real pipeline:

1. `create-dataset` (profile `v2.5-tabpfn-settings`)
2. `generate-candidates` / `sample_random_settings.py` (vary model+optimizer, GT via train.py)
3. Build settings table → 5-fold TabPFN / baselines

Script: `scripts/run_family_pipeline.py`

## Families

- univariate_regression (target `mean_test_mse`)
- multivariate_regression (target `mean_test_mse`)
- bigram_lm (target `mean_test_ce`)
- synthetic_tabular_classification (target `mean_test_ce`)

Profile uses `n_seeds: 3` for faster A100 turnaround.

## Remote

```bash
ssh root@10.210.22.136 -p 31178
cd /cephfs/renzirui/projects/ArchitectureIQ-tabpfn
source .venv-tabpfn/bin/activate
export PYTHONPATH=src
export TABPFN_SKIP_LICENSE=1 HF_ENDPOINT=https://hf-mirror.com
python experiments/tabpfn_settings/scripts/run_family_pipeline.py --count 40 --device cuda
```
