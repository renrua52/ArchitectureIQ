# TabPFN settings → metric (side project)

**Question:** If we randomly sample ArchitectureIQ candidate *settings*
(model / optimizer / loss / budget), run GT once, and fit TabPFN on
`(setting features → selection metric)`, can TabPFN predict the metric for
**new** held-out settings?

This folder is self-contained under `experiments/tabpfn_settings/`.

## Plan (summary)

See [`PLAN.md`](./PLAN.md).

## Quick start (local table build — no GPU)

```bash
# from repo root
python experiments/tabpfn_settings/scripts/build_table.py \
  --pack benchmark_releases/question_packs/xor-v2.5-100q-37b9da \
  --out experiments/tabpfn_settings/artifacts/xor_table.csv
```

## Evaluate (needs `tabpfn` + preferably CUDA)

```bash
pip install -r experiments/tabpfn_settings/requirements.txt
python experiments/tabpfn_settings/scripts/evaluate.py \
  --table experiments/tabpfn_settings/artifacts/xor_table.csv \
  --target mean_test_ce \
  --device cuda
```

## Remote A100

Work under `/cephfs/renzirui/projects/ArchitectureIQ-tabpfn` on
`root@10.210.22.136:31178`. Prefer China mirrors for pip / Hugging Face.
