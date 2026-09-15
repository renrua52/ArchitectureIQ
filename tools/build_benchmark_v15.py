"""Build a 500-question v1.5 benchmark: one dataset instance per question.

v1.5 sets question_generation.quality.max_questions_per_dataset = 1, so a
500-question benchmark needs 500 dataset instances. This driver owns everything
the generator deliberately does not: the family quota, the question-type mix,
the budget spread, and the family-option (input_dim / rule_family) spread.
Per the 2026-09-01 ruling, quotas belong to the caller, not to
questions/generator.py.

It reuses the canonical pipeline only -- create_dataset ->
generate_candidate_set (which runs ground truth by executing the generated
train.py) -> generate_questions -> write_prompt. No parallel training logic.

Quota (500 questions):
    univariate_regression              100
    multivariate_regression            100
    bigram_lm                          100
    synthetic_tabular_classification   100
    xor_classification                  50
    spiral_classification               50

Within each family, question type cycles architecture_only / optimizer_only /
mixed and the budget cycles 4096 / 8192 / 16384 / 32768 on a coprime stride, so
all twelve (type, budget) pairs appear evenly.

Resume-safe: an item whose question already exists in the ledger is skipped.

Usage:
    python tools/build_benchmark_v15.py --profile v1.5 --workers 30 --count 12
    python tools/build_benchmark_v15.py --profile v1.5 --pilot 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architecture_iq.paths import DATA_DIR  # noqa: E402

FAMILY_QUOTA = (
    ("univariate_regression", 100),
    ("multivariate_regression", 100),
    ("bigram_lm", 100),
    ("synthetic_tabular_classification", 100),
    ("xor_classification", 50),
    ("spiral_classification", 50),
)
QUESTION_TYPES = ("architecture_only", "optimizer_only", "mixed")
BUDGETS = (4096, 8192, 16384, 32768)
MV_INPUT_DIMS = (2, 3, 4, 5, 8)
TAB_INPUT_DIMS = (2, 4, 8, 16)
TAB_RULES = ("smooth_additive", "sparse_interaction", "piecewise_boundary")
FAMILY_SEED_BASE = {
    "univariate_regression": 100_000,
    "multivariate_regression": 200_000,
    "bigram_lm": 300_000,
    "synthetic_tabular_classification": 400_000,
    "xor_classification": 500_000,
    "spiral_classification": 600_000,
}

LEDGER = DATA_DIR / "benchmark_v15_ledger.jsonl"
CLAIMS = DATA_DIR / "benchmark_v15_claims"


def build_plan() -> list[dict[str, Any]]:
    """Deterministic item plan. Item ids are stable across runs (resume key)."""
    plan: list[dict[str, Any]] = []
    for family, quota in FAMILY_QUOTA:
        for i in range(quota):
            options: dict[str, Any] = {}
            if family == "multivariate_regression":
                options["input_dim"] = MV_INPUT_DIMS[(i // 3) % len(MV_INPUT_DIMS)]
            elif family == "synthetic_tabular_classification":
                options["input_dim"] = TAB_INPUT_DIMS[(i // 3) % len(TAB_INPUT_DIMS)]
                options["rule_family"] = TAB_RULES[(i // 4) % len(TAB_RULES)]
            elif family == "xor_classification":
                options["input_dim"] = TAB_INPUT_DIMS[(i // 3) % len(TAB_INPUT_DIMS)]
            plan.append(
                {
                    "item_id": f"{family}#{i:03d}",
                    "family": family,
                    "index": i,
                    "question_type": QUESTION_TYPES[i % len(QUESTION_TYPES)],
                    "budget": BUDGETS[i % len(BUDGETS)],
                    "family_options": options,
                    "seed_base": FAMILY_SEED_BASE[family] + i * 37,
                }
            )
    return plan


def _varying_axes(question_type: str) -> frozenset[str]:
    if question_type == "architecture_only":
        return frozenset({"model"})
    if question_type == "optimizer_only":
        return frozenset({"optimizer"})
    if question_type == "mixed":
        return frozenset({"model", "optimizer"})
    raise ValueError(f"Unsupported question type for v1.5: {question_type}")


def _claim(dataset_id: str) -> bool:
    """Atomically claim a dataset id so no two questions share one instance."""
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(CLAIMS / dataset_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def build_one(
    item: dict[str, Any],
    profile_name: str,
    count: int,
    ds_attempts: int,
    set_attempts: int,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from architecture_iq.datasets import create_dataset
    from architecture_iq.candidates.sets import generate_candidate_set
    from architecture_iq.profile import load_profile
    from architecture_iq.prompts.renderer import write_prompt
    from architecture_iq.questions.generator import generate_questions
    from architecture_iq.registry import ensure_registries, get_dataset_family

    ensure_registries()
    profile = load_profile(profile_name)
    family_name = item["family"]
    family = get_dataset_family(family_name)
    qtype = item["question_type"]
    varying = _varying_axes(qtype)
    started = time.time()
    notes: list[str] = []

    for ds_attempt in range(ds_attempts):
        ds_seed = item["seed_base"] + 1_009 * ds_attempt
        try:
            # Same two plugin calls create_dataset makes, so the id is known
            # before materializing and can be claimed atomically.
            probe = family.build_spec_with_id(
                family.create_instance(profile, ds_seed, **item["family_options"])
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ds_seed={ds_seed} create_instance failed: {exc!r}")
            continue
        dataset_id = probe["dataset_id"]
        if not _claim(dataset_id):
            notes.append(f"ds_seed={ds_seed} dataset_id {dataset_id} already claimed")
            continue
        try:
            _spec, dataset_path = create_dataset(
                profile,
                ds_seed,
                family_name=family_name,
                family_options=item["family_options"] or None,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ds_seed={ds_seed} create_dataset failed: {exc!r}")
            continue

        for set_attempt in range(set_attempts):
            set_seed = item["seed_base"] * 3 + 7_919 * set_attempt + 101 * ds_attempt
            set_count = count if set_attempt == 0 else max(6, count - 2 * set_attempt)
            try:
                set_path = generate_candidate_set(
                    profile,
                    dataset_path=dataset_path,
                    budget=item["budget"],
                    count=set_count,
                    varying_axes=varying,
                    rng=random.Random(set_seed),
                    seed=set_seed,
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"set_seed={set_seed} count={set_count} candidate set failed: {exc!r}"
                )
                continue
            try:
                run_path, results = generate_questions(
                    profile,
                    dataset_path=dataset_path,
                    candidate_set_paths=[set_path],
                    rng=random.Random(set_seed + 1),
                    num_questions=1,
                    num_choices=3,
                    seed=set_seed + 1,
                    question_type=qtype,
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"set_seed={set_seed} question assembly failed: {exc!r}")
                continue
            record, out = results[0]
            write_prompt(out)
            return {
                "item_id": item["item_id"],
                "status": "ok",
                "family": family_name,
                "question_type": record["type"],
                "requested_type": qtype,
                "budget": item["budget"],
                "family_options": item["family_options"],
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "set_path": str(set_path),
                "set_count": set_count,
                "run_path": str(run_path),
                "question_id": record["question_id"],
                "question_path": str(out),
                "correct_letter": record["correct_letter"],
                "varying_axes": record["varying_axes"],
                "ds_seed": ds_seed,
                "set_seed": set_seed,
                "seconds": round(time.time() - started, 1),
                "notes": notes,
            }

    return {
        "item_id": item["item_id"],
        "status": "failed",
        "family": family_name,
        "requested_type": qtype,
        "budget": item["budget"],
        "family_options": item["family_options"],
        "seconds": round(time.time() - started, 1),
        "notes": notes,
    }


def _worker(payload: tuple[dict[str, Any], str, int, int, int]) -> dict[str, Any]:
    item, profile_name, count, ds_attempts, set_attempts = payload
    try:
        return build_one(item, profile_name, count, ds_attempts, set_attempts)
    except Exception:  # noqa: BLE001
        return {
            "item_id": item["item_id"],
            "status": "crashed",
            "family": item["family"],
            "requested_type": item["question_type"],
            "budget": item["budget"],
            "traceback": traceback.format_exc()[-2000:],
        }


def load_ledger() -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not LEDGER.exists():
        return done
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        prev = done.get(rec["item_id"])
        if prev is None or rec["status"] == "ok":
            done[rec["item_id"]] = rec
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v1.5")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--count", type=int, default=12, help="candidates per set")
    ap.add_argument("--ds-attempts", type=int, default=3)
    ap.add_argument("--set-attempts", type=int, default=3)
    ap.add_argument("--pilot", type=int, default=0, help="take N items round-robin per family")
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--families", default=None, help="comma-separated family allow-list")
    ap.add_argument("--item", action="append", default=[], help="build only these item ids")
    ap.add_argument("--retry-failed", action="store_true", help="re-run items not marked ok")
    ap.add_argument(
        "--longest-first",
        action="store_true",
        help="submit large-budget items first so the slow tail overlaps the cheap ones",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = build_plan()
    if args.item:
        wanted = set(args.item)
        plan = [item for item in plan if item["item_id"] in wanted]
    if args.only_family:
        plan = [item for item in plan if item["family"] == args.only_family]
    if args.families:
        allow = {name.strip() for name in args.families.split(",") if name.strip()}
        plan = [item for item in plan if item["family"] in allow]
    if args.pilot:
        per_family: dict[str, int] = {}
        picked = []
        for item in plan:
            n = per_family.get(item["family"], 0)
            if n < args.pilot:
                picked.append(item)
                per_family[item["family"]] = n + 1
        plan = picked

    ledger = load_ledger()
    pending = []
    for item in plan:
        rec = ledger.get(item["item_id"])
        if rec is None:
            pending.append(item)
        elif rec["status"] != "ok" and args.retry_failed:
            pending.append(item)

    if args.longest_first:
        pending.sort(key=lambda item: -item["budget"])

    print(
        f"plan={len(plan)} already_ok={sum(1 for i in plan if ledger.get(i['item_id'], {}).get('status') == 'ok')} "
        f"pending={len(pending)} workers={args.workers} count={args.count}",
        flush=True,
    )
    if args.dry_run or not pending:
        for item in pending[:10]:
            print("  would build", item, flush=True)
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        (item, args.profile, args.count, args.ds_attempts, args.set_attempts)
        for item in pending
    ]
    ok = 0
    bad = 0
    t0 = time.time()
    with LEDGER.open("a") as ledger_fp, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
        for n, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            ledger_fp.write(json.dumps(rec, sort_keys=True) + "\n")
            ledger_fp.flush()
            if rec["status"] == "ok":
                ok += 1
            else:
                bad += 1
            elapsed = time.time() - t0
            rate = n / elapsed * 3600 if elapsed else 0
            print(
                f"[{n}/{len(payloads)}] {rec['item_id']} {rec['status']} "
                f"{rec.get('question_id', '')} {rec.get('seconds', '')}s "
                f"| ok={ok} bad={bad} {rate:.0f} items/h",
                flush=True,
            )
    print(f"done ok={ok} bad={bad} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
