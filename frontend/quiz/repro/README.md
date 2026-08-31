# ArchitectureIQ — reproducible question bundle

This folder is everything needed to re-run the ground truth for one benchmark
question on your own machine. Nothing is precomputed: the dataset is
synthesized from source, and each choice is trained by the same `train.py` that
produced the published result.

## Run it

```bash
pip install torch numpy      # a CPU-only torch build is enough
python reproduce.py          # every choice, every seed
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--seeds 1` | run only the first seed per choice (fast smoke test) |
| `--letters A,C` | run only some choices |
| `--tol 1e-3` | relative tolerance when comparing against the recorded reference |
| `--threads 4` | torch intra-op threads (default: the count the ground truth used) |
| `--device cuda` | run on a GPU instead of CPU (results will differ from the CPU reference) |
| `--save-tensors` | also dump the synthesized splits to `dataset/train.pt` / `test.pt` |

Cost: one choice × one seed is seconds to a minute; all choices × all seeds is
minutes on a laptop.

## What is in here

```
README.md              this file
reproduce.py           the runner (no ArchitectureIQ install required)
question.json          question metadata: metric, seeds, per-choice budgets
prompt.txt             the exact prompt text the benchmark shows
dataset/
  dataset_spec.json    frozen dataset parameters
  synthesize.py        generates the train/test splits — the source of truth
choices/<A|B|C>/
  candidate_spec.json  frozen model + optimizer + loss + budget
  model.py             the model, rendered from that spec
  loss.py              the loss
  optimizer.py         the optimizer
  train.py             the training loop; defines train_and_eval()
  reference/
    summary.json       the published per-seed and mean/std results
```

`reference/summary.json` and `question.json`'s `correct_letter` are present
**only if you downloaded this bundle after answering the question**. A bundle
downloaded beforehand contains the same code and lets you compute every number
yourself — it just does not ship the answer key.

Note that `question.json` here is a bundle-local convenience file. It is not the
pipeline's own `question.json`; it carries only what `reproduce.py` needs.

## How the ground truth is defined

1. `synthesize()` in `dataset/synthesize.py` materializes one fixed train/test
   split. Every choice sees exactly the same data.
2. Each choice is trained `n_seeds` times, with seeds
   `base_seed … base_seed + n_seeds - 1`, one `torch.manual_seed(seed)` per run.
3. A run's score is `final_<selection_metric>` on the full held-out test split
   at the last step.
4. A choice's score is the **mean** over seeds (`std` uses `ddof=0`). The choice
   with the lowest mean wins; that is `correct_letter`.

Each choice is scored at its own stated budget
(`training_steps × batch_size = total_samples_seen`). When budgets differ across
choices, that difference is part of the question.

## About exactness

With the same torch build and the same thread count, these runs reproduce the
recorded numbers bit for bit. Changing either one shifts results slightly —
float32 reductions are not associative, so a different number of threads sums
partial results in a different order. Typical drift is around `1e-7` relative,
far below the gap between choices, which is why `reproduce.py` compares with a
relative tolerance (default `1e-4`) rather than demanding equality.

`reproduce.py` reads the recorded thread count from the reference results and
sets it automatically when available. The torch version used for the published
run is in `choices/*/reference/summary.json` under `environment`.

Deviations far above the tolerance mean something real differs (a different
device, a modified file, a very different torch major version). `reproduce.py`
exits non-zero and lists every mismatch in that case.

## Questions

rzr23@mails.tsinghua.edu.cn
