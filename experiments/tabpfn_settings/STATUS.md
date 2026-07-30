# Status

## Done

- Branch: `side/tabpfn-settings` (from latest `main`)
- Code under `experiments/tabpfn_settings/`
- Feature encoder + XOR pack table (`artifacts/xor_table.csv`, 63 rows)
- P0 **baselines** (5-fold CV, target `mean_test_ce`):

| Model | MAE | RMSE | Spearman | Pairwise rank acc |
|-------|-----|------|----------|-------------------|
| train_mean | 0.034 | 0.051 | n/a | 0.54 |
| Ridge | 0.031 | 0.044 | 0.57 | 0.69 |
| HistGB | 0.028 | 0.043 | 0.54 | 0.68 |

- Remote clone: `/cephfs/renzirui/projects/ArchitectureIQ-tabpfn`
- Remote `uv` env created; TabPFN 8.2 installed once
- First `torch` wheel was **cu130** → `cuda=False` on driver CUDA 12.9
- Reinstall of `torch==2.5.1+cu124` started; then **SSH to 10.210.22.136:31178 timed out**

## Next (when machine is reachable)

```bash
ssh root@10.210.22.136 -p 31178
cd /cephfs/renzirui/projects/ArchitectureIQ-tabpfn
source .venv-tabpfn/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_LINK_MODE=copy
# finish torch cu124 if needed:
uv pip install --link-mode=copy 'torch==2.5.1+cu124' --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; assert torch.cuda.is_available()"
python experiments/tabpfn_settings/scripts/evaluate.py --device cuda
```

Then P1: sample random settings via `scripts/sample_random_settings.py` on the XOR dataset and re-evaluate at larger N.
