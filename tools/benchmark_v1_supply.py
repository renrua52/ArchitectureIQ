"""V1-final (v1.1) benchmark supply generator: datasets + candidate-set skeletons.

Writes dataset instances and candidate sets (specs + .py files + set manifests)
under ``data/datasets/`` WITHOUT running ground truth. GT is done separately by
``tools/benchmark_v1_gt.py``. Reuses the canonical pipeline only
(``create_dataset`` / ``sample_candidate_set_pool`` / ``write_candidate`` /
``write_set_manifest``) — no parallel logic.

Supply design (v1.1, confirmed in docs/plans/v1-final-1000q-plan.md):
- 43 dataset instances: 7 per bucket (univariate / multivariate / bigram / xor /
  spiral) + 8 general_tabular (smooth_additive 3, sparse_interaction 3,
  piecewise_boundary 2). Buckets match benchmark_v1_build.DATASET_BUCKETS.
- 8 sets per dataset (344 total), vary-pattern mix tuned to the old question
  quotas (arch 400 / opt 300 / mixed 300):
  3x {model}, 2x {optimizer}, 2x {model,optimizer,loss}, 1x {optimizer,loss}.
- Budgets cycle the four v1.1 tiers [2048, 4096, 8192, 16384].
- 12-16 candidates per set (Q10: increased supply to reduce relaxed questions).

Idempotent: existing dataset instances and set dirs are skipped.

Usage:
    python tools/benchmark_v1_supply.py --profile v1.1 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architecture_iq.candidates.generator import write_candidate
from architecture_iq.candidates.sets import (
    make_set_name,
    sample_candidate_set_pool,
    write_set_manifest,
)
from architecture_iq.datasets import create_dataset
from architecture_iq.paths import DATA_DIR, candidate_set_dir, dataset_dir
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type

BUDGET_TIERS = (2048, 4096, 8192, 16384)

# (family, rule_family, count, seed_base) — rule_family only for stabcls.
DATASET_PLAN = [
    ("univariate_regression", None, 7, 1000),
    ("multivariate_regression", None, 7, 2000),
    ("bigram_lm", None, 7, 3000),
    ("synthetic_tabular_classification", "xor", 7, 4000),
    ("synthetic_tabular_classification", "spiral", 7, 5000),
    ("synthetic_tabular_classification", "smooth_additive", 3, 6000),
    ("synthetic_tabular_classification", "sparse_interaction", 3, 6100),
    ("synthetic_tabular_classification", "piecewise_boundary", 2, 6200),
]

# 8 sets per dataset; aligned with question-type quotas arch/opt/mixed ~ 4/3/3.
VARY_PATTERNS = [
    frozenset({"model"}),
    frozenset({"model"}),
    frozenset({"model"}),
    frozenset({"optimizer"}),
    frozenset({"optimizer"}),
    frozenset({"model", "optimizer", "loss"}),
    frozenset({"model", "optimizer", "loss"}),
    frozenset({"optimizer", "loss"}),
]

INDEX_PATH = DATA_DIR / "benchmark_v1_supply_index.json"

# Top-up recipe (2026-08-23, user decision): enlarge arch sets to n=30 for all
# buckets (fills param_similar strata; bigram gap-constrained shortfall) and
# mixed sets to n=30 for the three stabcls buckets (xor short + thin margins).
TOPUP_ARCH_PER_DATASET = 3
TOPUP_MIXED_PER_DATASET = 2  # synthetic_tabular_classification datasets only
TOPUP_COUNT = 30
TOPUP_ARCH_VARY = frozenset({"model"})
TOPUP_MIXED_VARY = frozenset({"model", "optimizer", "loss"})


def _create_dataset_idempotent(profile, family_name, seed, rule_family):
    """Create a dataset instance, skipping if the content-addressed dir exists."""
    family = get_dataset_family(family_name)
    options = {"rule_family": rule_family} if rule_family is not None else None
    if options is not None:
        partial = family.create_instance(profile, seed, **options)
    else:
        partial = family.create_instance(profile, seed)
    spec = family.build_spec_with_id(partial)
    out = dataset_dir(family.name, spec["dataset_id"])
    if (out / "dataset_spec.json").is_file():
        return "skip", json.loads((out / "dataset_spec.json").read_text()), out
    # Not present: go through the canonical create path (materialize runs
    # synthesize.py). create_dataset re-derives the same spec from the seed.
    created_spec, out = create_dataset(
        profile, seed, family_name=family_name, family_options=options
    )
    return "created", created_spec, out


def _write_set_skeleton(profile, dataset_path, dataset_spec, vary, budget, count, seed):
    """Sample specs + write candidate files + set manifest (no GT)."""
    rng = random.Random(seed)
    specs = sample_candidate_set_pool(
        profile,
        dataset_id=dataset_spec["dataset_id"],
        family=dataset_spec["family"],
        budget=budget,
        count=count,
        varying_axes=vary,
        rng=rng,
        fixed_shared=None,
        dataset_params=dataset_spec["params"],
    )
    set_name = make_set_name(budget, vary, salt=rng.randint(0, 2**31 - 1))
    set_path = candidate_set_dir(dataset_path, set_name)
    if set_path.exists():
        return set_name, set_path, 0, "skip"

    shared_record = {}
    if specs:
        shared_record["batch_size"] = specs[0]["budget"]["batch_size"]
        if "model" not in vary:
            shared_record["model"] = specs[0]["model"]
        if "optimizer" not in vary:
            shared_record["optimizer"] = specs[0]["optimizer"]
        if "loss" not in vary:
            shared_record["loss"] = specs[0]["loss"]

    set_path.mkdir(parents=True, exist_ok=False)
    write_set_manifest(
        set_path,
        set_name=set_name,
        budget=budget,
        count=count,
        varying_axes=vary,
        fixed_shared=shared_record,
        model_type_counts=None,
        seed=seed,
        profile=profile,
        dataset_id=dataset_spec["dataset_id"],
        family=dataset_spec["family"],
    )
    written = 0
    for spec in specs:
        out = set_path / spec["candidate_id"]
        if out.exists():
            continue  # content-addressed duplicate within the pool
        write_candidate(spec, out, get_model_type(spec["model"]["type"]))
        written += 1
    return set_name, set_path, written, "created"


def _existing_big_sets(dataset_path: Path, vary: frozenset[str]) -> int:
    """How many sets of this vary pattern with count >= TOPUP_COUNT exist."""
    n = 0
    for manifest_path in (dataset_path / "candidates").glob("*/set.json"):
        manifest = json.loads(manifest_path.read_text())
        if (
            frozenset(manifest["varying_axes"]) == vary
            and int(manifest["count"]) >= TOPUP_COUNT
        ):
            n += 1
    return n


def _run_topup(profile, dry_run: bool) -> None:
    import zlib

    n_sets = n_cands = 0
    dataset_specs = sorted(DATA_DIR.glob("datasets/*/*/dataset_spec.json"))
    for spec_path in dataset_specs:
        dataset_spec = json.loads(spec_path.read_text())
        dataset_path = spec_path.parent
        family = dataset_spec["family"]
        base = zlib.crc32(dataset_spec["dataset_id"].encode()) % 100000

        recipes = [(TOPUP_ARCH_VARY, TOPUP_ARCH_PER_DATASET)]
        if family == "synthetic_tabular_classification":
            recipes.append((TOPUP_MIXED_VARY, TOPUP_MIXED_PER_DATASET))

        for vary, target in recipes:
            missing = target - _existing_big_sets(dataset_path, vary)
            for si in range(max(0, missing)):
                seed = base + 5000 + si * 13 + (0 if vary == TOPUP_ARCH_VARY else 700)
                budget = BUDGET_TIERS[(base + si) % len(BUDGET_TIERS)]
                label = f"{dataset_spec['dataset_id']} vary={sorted(vary)} #{si}"
                if dry_run:
                    print(f"[dry] topup set {label} budget={budget} n={TOPUP_COUNT}")
                    continue
                set_name, set_path, written, status = _write_set_skeleton(
                    profile, dataset_path, dataset_spec, vary, budget,
                    TOPUP_COUNT, seed,
                )
                if status == "created":
                    n_sets += 1
                    n_cands += written
                print(f"[topup {status}] {label} -> {set_name} "
                      f"budget={budget} candidates={written}", flush=True)

    print(f"[topup done] sets created={n_sets} candidates={n_cands}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v1.1")
    ap.add_argument("--topup", action="store_true",
                    help="Add n=30 arch sets (all datasets) and n=30 mixed sets "
                         "(stabcls datasets) to fill param_similar strata.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_registries()
    profile = load_profile(args.profile)

    if args.topup:
        _run_topup(profile, args.dry_run)
        return

    n_ds_created = n_ds_skip = n_sets_created = n_sets_skip = n_cands = 0
    index: list[dict] = []

    for family, rule_family, count, seed_base in DATASET_PLAN:
        for i in range(count):
            seed = seed_base + i
            label = f"{family}:{rule_family or '-'}#{i}"
            if args.dry_run:
                print(f"[dry] dataset {label} seed={seed}")
                continue
            status, dataset_spec, dataset_path = _create_dataset_idempotent(
                profile, family, seed, rule_family
            )
            if status == "created":
                n_ds_created += 1
            else:
                n_ds_skip += 1
            print(f"[dataset {status}] {label} -> {dataset_path.name}", flush=True)

            for si, vary in enumerate(VARY_PATTERNS):
                budget = BUDGET_TIERS[si % len(BUDGET_TIERS)]
                cand_count = 12 + (seed + si * 7) % 5  # 12..16
                set_seed = seed * 100 + si
                set_name, set_path, written, s_status = _write_set_skeleton(
                    profile, dataset_path, dataset_spec, vary, budget,
                    cand_count, set_seed,
                )
                if s_status == "created":
                    n_sets_created += 1
                    n_cands += written
                else:
                    n_sets_skip += 1
                index.append(
                    {
                        "dataset_path": str(dataset_path),
                        "set_path": str(set_path),
                        "set_name": set_name,
                        "family": family,
                        "rule_family": rule_family,
                        "vary": sorted(vary),
                        "budget": budget,
                        "n_candidates": written if s_status == "created" else None,
                    }
                )
                print(
                    f"  [set {s_status}] {set_name} vary={sorted(vary)} "
                    f"budget={budget} candidates={written}",
                    flush=True,
                )

    if args.dry_run:
        total_ds = sum(c for _, _, c, _ in DATASET_PLAN)
        print(f"[dry] would create up to {total_ds} datasets x "
              f"{len(VARY_PATTERNS)} sets = {total_ds * len(VARY_PATTERNS)} sets")
        return

    INDEX_PATH.write_text(json.dumps({"profile": args.profile, "sets": index}, indent=2))
    print(
        f"[done] datasets created={n_ds_created} skipped={n_ds_skip}; "
        f"sets created={n_sets_created} skipped={n_sets_skip}; "
        f"candidates written={n_cands}; index -> {INDEX_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
