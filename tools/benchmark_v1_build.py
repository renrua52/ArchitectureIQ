#!/usr/bin/env python3
"""Build a 1000-question V1 benchmark from generated candidates + GT."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import choices_compatible, infer_axes
from architecture_iq.profile import load_profile
from architecture_iq.questions.generator import (
    _budget_field,
    build_question_record,
    load_candidate_pool_from_sets,
)
from architecture_iq.questions.quality import QuestionQualityFilters
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, short_hash, write_json


def qtype_bucket(invariant_axes: list[str], varying_axes: list[str]) -> str:
    training = {"model", "optimizer", "loss"}
    varying = set(varying_axes) & training
    if varying == {"model"}:
        return "architecture_only"
    if varying == {"optimizer"}:
        return "optimizer_only"
    if varying == {"loss"}:
        return "loss_only"
    return "mixed"


def target_bucket(specs: list[dict[str, Any]]) -> str:
    _, varying = infer_axes(specs)
    return qtype_bucket([], varying)


def load_subpool(set_path: Path, metric: str, *, max_failed: int) -> list[Path]:
    filters = QuestionQualityFilters(require_finite_mean=True, max_failed_seeds=max_failed)
    return load_candidate_pool_from_sets([set_path], filters=filters, selection_metric=metric)


def gap_threshold(combos: list[dict[str, Any]], metric: str, k: float) -> float:
    # k * max(std_metric) within the subset, computed per candidate set using GT stds
    # We pass k directly and let validate_significance compare against per-set sigma via summaries.
    # validate_significance uses absolute gap, so we pick candidate sets and compute max std.
    return k  # marker, actual sigma is computed in selection below


def collect_significant(
    paths: list[Path],
    profile,
    rng: random.Random,
    metric: str,
    *,
    n_choices: int,
    gap_cap: float | None,
    param_similar: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    summaries = {p: load_summary(p) for p in paths}
    specs = {p: read_json(p / "candidate_spec.json") for p in paths}
    params = {p: int(specs[p].get("trainable_parameter_count", 0)) for p in paths}

    for combo in combinations(paths, n_choices):
        specs_combo = [specs[p] for p in combo]
        if not choices_compatible(specs_combo):
            continue
        sums = [summaries[p] for p in combo]
        if any(s.get("excluded") for s in sums):
            continue
        sig = validate_significance(sums, profile, metric=metric)
        if not sig.passed:
            continue
        # gap_cap is absolute k*max(std) across summaries
        if gap_cap is not None:
            max_std = max(float(s.get(f"std_{metric}", 0.0)) for s in sums)
            if sig.gap > gap_cap * max_std:
                continue
        pcs = [params[p] for p in combo]
        if param_similar and max(pcs) > 1.5 * max(1, min(pcs)):
            continue
        out.append({"paths": list(combo), "sig": sig, "specs": specs_combo})
    rng.shuffle(out)
    return out


def benchmark_metadata(
    *,
    family: str,
    dataset_id: str,
    budget: int,
    bucket: str,
    gap_constrained: bool,
    param_similar: bool,
    specs: list[dict[str, Any]],
    sig_gap: float,
) -> dict[str, Any]:
    pcs = [int(s.get("trainable_parameter_count", 0)) for s in specs]
    bs = [int(s["budget"]["batch_size"]) for s in specs]
    return {
        "family": family,
        "dataset_id": dataset_id,
        "budget_tier": budget,
        "question_type": bucket,
        "gap_constrained": gap_constrained,
        "param_similar": param_similar,
        "gap": sig_gap,
        "batch_sizes": bs,
        "trainable_parameter_counts": pcs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/v1_llm"))
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--num-choices", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-k", type=float, default=5.0)
    parser.add_argument("--max-failed-seeds", type=int, default=1)
    parser.add_argument("--profiles", default="v1")
    parser.add_argument("--dry-run", action="store_true", help="Only report quotas/plan, do not write")
    args = parser.parse_args()

    profile = load_profile(args.profiles)
    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    out = args.out.resolve()

    sets_root = data_root / "datasets"
    if not sets_root.is_dir():
        raise SystemExit(f"missing datasets root: {sets_root}")

    # discover candidate sets grouped by (family, dataset_id, budget, set_id)
    groups: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for set_manifest in sets_root.glob("*/*/candidates/*/set.json"):
        manifest = read_json(set_manifest)
        family = manifest["family"]
        dataset_id = manifest["dataset_id"]
        budget = int(manifest["budget"]["total_samples_seen"])
        set_id = manifest["set_id"]
        dataset_dir = set_manifest.parents[2]
        groups[(family, dataset_id, budget, set_id)] = {
            "set_path": set_manifest.parent,
            "dataset_path": dataset_dir,
            "manifest": manifest,
        }

    if not groups:
        raise SystemExit("no candidate sets found")

    # quotas across benchmark buckets
    plan = []
    buckets = [
        ("architecture_only", 0.40),
        ("optimizer_only", 0.30),
        ("mixed", 0.30),
    ]
    for bucket, frac in buckets:
        n = int(round(args.num_questions * frac))
        n_gap = int(round(n * 0.70))
        n_ps = int(round(n * 0.80))
        plan.append({"question_type": bucket, "total": n, "gap_constrained": n_gap, "param_similar": n_ps})

    if args.dry_run:
        print(json.dumps({"groups": len(groups), "plan": plan}, indent=2))
        return

    selected: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()

    for target in plan:
        bucket = target["question_type"]
        needed_gap = target["gap_constrained"]
        needed_no_gap = target["total"] - target["gap_constrained"]
        for gap_constrained, remaining in [(True, needed_gap), (False, needed_no_gap)]:
            ps_needed = int(round(remaining * 0.80))
            candidates_for_bucket: list[dict[str, Any]] = []
            for (family, dataset_id, budget, set_id), info in sorted(groups.items(), key=lambda kv: kv[0]):
                dataset_spec = read_json(info["dataset_path"] / "dataset_spec.json")
                metric = dataset_spec["selection_metric"]
                pool = load_subpool(info["set_path"], metric, max_failed=args.max_failed_seeds)
                if len(pool) < args.num_choices:
                    continue
                specs_pool = {p: read_json(p / "candidate_spec.json") for p in pool}
                # partition by gap & param-similarity; note param_similar flips half-way below
                for param_similar in ([True, False] if remaining > 0 else []):
                    subsets = collect_significant(
                        pool, profile, rng, metric,
                        n_choices=args.num_choices,
                        gap_cap=args.gap_k if gap_constrained else None,
                        param_similar=param_similar,
                    )
                    for entry in subsets:
                        specs = entry["specs"]
                        if target_bucket(specs) != bucket:
                            continue
                        ids = frozenset(specs[i]["candidate_id"] for i in range(len(specs)))
                        if ids & seen_candidates:
                            continue
                        meta = benchmark_metadata(
                            family=family, dataset_id=dataset_id, budget=budget,
                            bucket=bucket, gap_constrained=gap_constrained,
                            param_similar=param_similar, specs=specs, sig_gap=entry["sig"].gap,
                        )
                        candidates_for_bucket.append({"info": info, "paths": entry["paths"], "meta": meta})
                        if len(candidates_for_bucket) >= remaining * 4:
                            break
                    if len(candidates_for_bucket) >= remaining * 4:
                        break
                if len(candidates_for_bucket) >= remaining * 4:
                    break
            # pick respecting param_similar split
            rng.shuffle(candidates_for_bucket)
            ps = [c for c in candidates_for_bucket if c["meta"]["param_similar"]]
            ns = [c for c in candidates_for_bucket if not c["meta"]["param_similar"]]
            take_ps = min(ps_needed, len(ps))
            take_ns = min(remaining - take_ps, len(ns))
            take = ps[:take_ps] + ns[:take_ns]
            if len(take) < remaining:
                extra = [c for c in candidates_for_bucket if c not in take]
                take += extra[: remaining - len(take)]
            for item in take:
                seen_candidates.update(p.name for p in item["paths"])
            selected.extend(take)

    if len(selected) < args.num_questions:
        raise SystemExit(f"only selected {len(selected)} questions; need more candidates or relax filters")

    out.mkdir(parents=True, exist_ok=True)
    collection = {
        "collection_id": f"v1_llm_{short_hash({'n': args.num_questions, 'seed': args.seed})}",
        "ordered": False,
        "question_paths": [],
        "records": [],
    }
    qdir = out / "questions"
    qdir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    for idx, item in enumerate(selected):
        info = item["info"]
        dataset_path = info["dataset_path"]
        dataset_spec = read_json(dataset_path / "dataset_spec.json")
        record = build_question_record(
            profile,
            dataset_spec=dataset_spec,
            dataset_path=dataset_path,
            candidate_paths=item["paths"],
            candidate_set_paths=[info["set_path"]],
            rng=rng,
            quality=QuestionQualityFilters(require_finite_mean=True, max_failed_seeds=args.max_failed_seeds),
            artifact_root=data_root,
            benchmark_metadata=item["meta"],
        )
        qid = record["question_id"]
        qpath = qdir / qid
        qpath.mkdir(parents=True, exist_ok=True)
        record["question_run_id"] = "benchmark_v1_llm"
        record["question_run_path"] = str(qpath.relative_to(data_root))
        write_json(qpath / "question.json", record)
        collection["question_paths"].append(str(qpath.relative_to(data_root)))
        collection["records"].append({"question_id": qid, "order": idx, **item["meta"]})
        stats[("family", item["meta"]["family"])] += 1
        stats[("question_type", item["meta"]["question_type"])] += 1
        stats[("budget_tier", item["meta"]["budget_tier"])] += 1
        stats[("gap_constrained", item["meta"]["gap_constrained"])] += 1
        stats[("param_similar", item["meta"]["param_similar"])] += 1

    manifest = {
        "name": "v1_llm",
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "num_questions": len(selected),
        "num_choices": args.num_choices,
        "seed": args.seed,
        "gap_k": args.gap_k,
        "max_failed_seeds": args.max_failed_seeds,
        "selection_metric_by_family": {
            family: read_json(info["dataset_path"] / "dataset_spec.json")["selection_metric"]
            for (family, _d, _b, _s), info in list(groups.items())[:10]
        },
        "stats": {f"{k[0]}:{k[1]}": v for k, v in stats.items()},
        "collection": collection["collection_id"],
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "collection.json", collection)
    print(json.dumps({"questions": len(selected), "manifest": str(out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
