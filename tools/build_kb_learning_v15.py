#!/usr/bin/env python3
"""Build fresh v1.5 learning questions without touching the fixed benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_benchmark_v15 as benchmark_builder  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LEDGER = DATA_DIR / "kb_learning_v15_ledger.jsonl"
CLAIMS = DATA_DIR / "kb_learning_v15_claims"
EPOCHS_DIR = DATA_DIR / "kb_learning_v15_epochs"
FAMILIES = tuple(family for family, _quota in benchmark_builder.FAMILY_QUOTA)
LEARNING_SEED_BASE = {
    family: 10_000_000 + index * 1_000_000
    for index, family in enumerate(FAMILIES, start=1)
}


def build_epoch_plan(epoch: int) -> list[dict[str, Any]]:
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    family_counts = {family: 0 for family in FAMILIES}
    plan: list[dict[str, Any]] = []
    for slot in range(50):
        family = FAMILIES[(slot + epoch - 1) % len(FAMILIES)]
        local_index = family_counts[family]
        family_counts[family] += 1
        global_index = (epoch - 1) * 9 + local_index
        options: dict[str, Any] = {}
        if family == "multivariate_regression":
            options["input_dim"] = benchmark_builder.MV_INPUT_DIMS[
                global_index % len(benchmark_builder.MV_INPUT_DIMS)
            ]
        elif family == "synthetic_tabular_classification":
            options["input_dim"] = benchmark_builder.TAB_INPUT_DIMS[
                global_index % len(benchmark_builder.TAB_INPUT_DIMS)
            ]
            options["rule_family"] = benchmark_builder.TAB_RULES[
                global_index % len(benchmark_builder.TAB_RULES)
            ]
        elif family == "xor_classification":
            options["input_dim"] = benchmark_builder.TAB_INPUT_DIMS[
                global_index % len(benchmark_builder.TAB_INPUT_DIMS)
            ]
        plan.append(
            {
                "item_id": f"kb-e{epoch:04d}-{family}-{local_index:02d}",
                "family": family,
                "index": global_index,
                "question_type": benchmark_builder.QUESTION_TYPES[
                    (slot + epoch - 1) % len(benchmark_builder.QUESTION_TYPES)
                ],
                "budget": benchmark_builder.BUDGETS[
                    (slot + 2 * epoch - 2) % len(benchmark_builder.BUDGETS)
                ],
                "family_options": options,
                "seed_base": LEARNING_SEED_BASE[family] + epoch * 10_000 + local_index * 37,
            }
        )
    return plan


def load_ledger() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not LEDGER.is_file():
        return records
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["item_id"]] = record
    return records


def link_question(epoch: int, record: dict[str, Any]) -> None:
    if record.get("status") != "ok":
        return
    target = Path(record["question_path"]).resolve()
    link = EPOCHS_DIR / f"epoch_{epoch:04d}" / str(record["question_id"])
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise FileExistsError(f"Question link points elsewhere: {link}")
        return
    link.symlink_to(target, target_is_directory=True)


def _worker(payload: tuple[dict[str, Any], str, int, int, int]) -> dict[str, Any]:
    item, profile, count, ds_attempts, set_attempts = payload
    try:
        benchmark_builder.CLAIMS = CLAIMS
        return benchmark_builder.build_one(item, profile, count, ds_attempts, set_attempts)
    except Exception:  # noqa: BLE001
        return {
            "item_id": item["item_id"],
            "status": "crashed",
            "family": item["family"],
            "requested_type": item["question_type"],
            "budget": item["budget"],
            "traceback": traceback.format_exc()[-2000:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--profile", default="v1.5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--ds-attempts", type=int, default=3)
    parser.add_argument("--set-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_epoch_plan(args.epoch)
    ledger = load_ledger()
    for item in plan:
        record = ledger.get(item["item_id"])
        if record and record.get("status") == "ok":
            link_question(args.epoch, record)
    pending = [
        item
        for item in plan
        if item["item_id"] not in ledger
        or (args.retry_failed and ledger[item["item_id"]].get("status") != "ok")
    ]
    print(
        f"epoch={args.epoch} plan={len(plan)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        for item in pending:
            print(json.dumps(item, sort_keys=True))
        return 0
    if not pending:
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        (item, args.profile, args.count, args.ds_attempts, args.set_attempts)
        for item in pending
    ]
    completed = 0
    failed = 0
    started = time.time()
    with LEDGER.open("a", encoding="utf-8") as ledger_file:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                record = future.result()
                ledger_file.write(json.dumps(record, sort_keys=True) + "\n")
                ledger_file.flush()
                if record["status"] == "ok":
                    completed += 1
                    link_question(args.epoch, record)
                else:
                    failed += 1
                done = completed + failed
                print(
                    f"[{done}/{len(pending)}] {record['item_id']} {record['status']} "
                    f"{record.get('seconds', '')}s | ok={completed} failed={failed}",
                    flush=True,
                )
    print(
        f"finished epoch={args.epoch} ok={completed} failed={failed} "
        f"minutes={(time.time() - started) / 60:.1f}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
