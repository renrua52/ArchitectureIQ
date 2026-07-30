#!/usr/bin/env python3
"""Build a 1000-question V1 LLM benchmark from generated candidates + GT.

Stratified assembly:
- 6 dataset buckets (~1/6 each): univariate / multivariate / bigram / xor /
  spiral / general tabular classification.
- Question types per bucket: 40% architecture_only, 30% optimizer_only, 30% mixed.
- 70% gap-constrained (winner–runner-up gap <= gap_k * max std_metric), 30% unrestricted.
- 80% param-capped (trainable_parameter_count max/min <= param_ratio_cap), 20% unrestricted.
- Budget tiers 1024/2048/4096/8192/16384 balanced greedily across selections.

Every question uses 3 choices from a single candidate set (shared
total_samples_seen), passes pool rules (require_finite_mean,
max_failed_seeds) and validate_significance.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import choices_compatible, infer_axes
from architecture_iq.paths import ROOT
from architecture_iq.profile import load_profile
from architecture_iq.prompts.renderer import write_prompt
from architecture_iq.questions.generator import (
    build_question_record,
    load_candidate_pool_from_sets,
)
from architecture_iq.questions.quality import QuestionQualityFilters
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, short_hash, write_json

DATASET_BUCKETS = ("univariate", "multivariate", "bigram", "xor", "spiral", "general_tabular")
QUESTION_TYPES = ("architecture_only", "optimizer_only", "mixed")
BUDGET_TIERS = (1024, 2048, 4096, 8192, 16384)


def dataset_bucket(family: str, dataset_spec: dict[str, Any]) -> str:
    if family == "univariate_regression":
        return "univariate"
    if family == "multivariate_regression":
        return "multivariate"
    if family == "bigram_lm":
        return "bigram"
    if family == "synthetic_tabular_classification":
        rule = str(dataset_spec.get("params", {}).get("rule_family", ""))
        if rule == "xor":
            return "xor"
        if rule == "spiral":
            return "spiral"
        return "general_tabular"
    raise ValueError(f"Unknown family for benchmark bucketing: {family}")


def qtype_of(specs: list[dict[str, Any]]) -> str:
    _, varying = infer_axes(specs)
    training = {"model", "optimizer", "loss"}
    v = set(varying) & training
    if v == {"model"}:
        return "architecture_only"
    if v == {"optimizer"}:
        return "optimizer_only"
    if v == {"loss"}:
        return "loss_only"
    return "mixed"


def build_plan(num_questions: int) -> list[dict[str, Any]]:
    """72-stratum plan: bucket x type x gap x param quotas."""
    base = num_questions // len(DATASET_BUCKETS)
    rem = num_questions - base * len(DATASET_BUCKETS)
    plan: list[dict[str, Any]] = []
    for bi, bucket in enumerate(DATASET_BUCKETS):
        n_b = base + (1 if bi < rem else 0)
        n_arch = int(round(n_b * 0.40))
        n_opt = int(round(n_b * 0.30))
        n_mix = n_b - n_arch - n_opt
        for qtype, n_t in (("architecture_only", n_arch), ("optimizer_only", n_opt), ("mixed", n_mix)):
            n_gap = int(round(n_t * 0.70))
            n_nogap = n_t - n_gap
            for gap_constrained, n_g in ((True, n_gap), (False, n_nogap)):
                n_ps = int(round(n_g * 0.80))
                n_ns = n_g - n_ps
                for param_similar, quota in ((True, n_ps), (False, n_ns)):
                    plan.append(
                        {
                            "dataset_bucket": bucket,
                            "question_type": qtype,
                            "gap_constrained": gap_constrained,
                            "param_similar": param_similar,
                            "quota": quota,
                            "filled": 0,
                            "relaxed_filled": 0,
                        }
                    )
    return plan


def index_set_entries(
    set_path: Path,
    pool: list[Path],
    profile,
    metric: str,
    *,
    n_choices: int,
    gap_k: float,
    param_ratio_cap: float,
    dataset_bucket_name: str,
) -> list[dict[str, Any]]:
    """Enumerate all significant 3-choice combos of a set with their properties."""
    summaries = {p: load_summary(p) for p in pool}
    specs = {p: read_json(p / "candidate_spec.json") for p in pool}
    entries: list[dict[str, Any]] = []
    for combo in combinations(pool, n_choices):
        specs_combo = [specs[p] for p in combo]
        if not choices_compatible(specs_combo):
            continue
        sums = [summaries[p] for p in combo]
        if any(s.get("excluded") for s in sums):
            continue
        sig = validate_significance(sums, profile, metric=metric)
        if not sig.passed:
            continue
        max_std = max(float(s.get(f"std_{metric}", 0.0)) for s in sums)
        gap_cap = gap_k * max_std
        gap_ok = sig.gap <= gap_cap
        pcs = [int(s.get("trainable_parameter_count", 0)) for s in specs_combo]
        ratio = max(pcs) / max(1, min(pcs))
        ps_ok = ratio <= param_ratio_cap
        model_types = [str(s["model"]["type"]) for s in specs_combo]
        entries.append(
            {
                "paths": list(combo),
                "question_type": qtype_of(specs_combo),
                "gap_ok": gap_ok,
                "ps_ok": ps_ok,
                "gap": sig.gap,
                "gap_cap": gap_cap,
                "param_ratio": ratio,
                "trainable_parameter_counts": pcs,
                "batch_sizes": [int(s["budget"]["batch_size"]) for s in specs_combo],
                "budget": int(specs_combo[0]["budget"]["total_samples_seen"]),
                "dataset_bucket": dataset_bucket_name,
                "model_types": model_types,
                "winner_model_type": model_types[sig.winner_index],
                "win_rate": sig.win_rate,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/v1_llm"))
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--num-choices", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-k", type=float, default=5.0)
    parser.add_argument("--param-ratio-cap", type=float, default=1.5)
    parser.add_argument("--max-failed-seeds", type=int, default=1)
    parser.add_argument("--profiles", default="v1")
    parser.add_argument("--no-prompts", action="store_true", help="Skip prompt.txt rendering")
    parser.add_argument("--allow-reuse-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only report quotas/plan, do not write")
    args = parser.parse_args()

    profile = load_profile(args.profiles)
    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    out = args.out.resolve()

    sets_root = data_root / "datasets"
    if not sets_root.is_dir():
        raise SystemExit(f"missing datasets root: {sets_root}")

    groups: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for set_manifest in sets_root.glob("*/*/candidates/*/set.json"):
        manifest = read_json(set_manifest)
        family = manifest["family"]
        dataset_id = manifest["dataset_id"]
        budget = int(manifest["budget"]["total_samples_seen"])
        set_id = manifest["set_id"]
        groups[(family, dataset_id, budget, set_id)] = {
            "set_path": set_manifest.parent,
            "dataset_path": set_manifest.parents[2],
            "manifest": manifest,
        }
    if not groups:
        raise SystemExit("no candidate sets found")

    filters = QuestionQualityFilters(require_finite_mean=True, max_failed_seeds=args.max_failed_seeds)

    # ---- index all significant combos per set, grouped by dataset bucket ----
    bucket_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gt_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[assignment]
    selection_metric_by_family: dict[str, str] = {}
    for (family, dataset_id, budget, set_id), info in sorted(groups.items(), key=lambda kv: kv[0]):
        dataset_spec = read_json(info["dataset_path"] / "dataset_spec.json")
        metric = dataset_spec["selection_metric"]
        selection_metric_by_family.setdefault(family, metric)
        bucket = dataset_bucket(family, dataset_spec)
        all_cands = [p for p in info["set_path"].glob("c_*") if (p / "candidate_spec.json").exists()]
        with_summary = [p for p in all_cands if (p / "results" / "summary.json").exists()]
        pool = load_candidate_pool_from_sets([info["set_path"]], filters=filters, selection_metric=metric)
        gt_stats[family]["candidates_total"] += len(all_cands)
        gt_stats[family]["with_summary"] += len(with_summary)
        gt_stats[family]["eligible"] += len(pool)
        gt_stats[family]["dropped_quality"] += len(with_summary) - len(pool)
        if len(pool) < args.num_choices:
            continue
        entries = index_set_entries(
            info["set_path"],
            pool,
            profile,
            metric,
            n_choices=args.num_choices,
            gap_k=args.gap_k,
            param_ratio_cap=args.param_ratio_cap,
            dataset_bucket_name=bucket,
        )
        rng.shuffle(entries)
        bucket_entries[bucket].extend(entries)

    plan = build_plan(args.num_questions)
    supply = {
        b: len(bucket_entries.get(b, [])) for b in DATASET_BUCKETS
    }
    if args.dry_run:
        print(json.dumps({"groups": len(groups), "supply": supply, "plan": plan}, indent=2))
        return

    # ---- stratified selection; constrained slots first, unrestricted from leftovers ----
    tier_counts: dict[int, int] = defaultdict(int)
    seen_candidates: set[str] = set()
    picked_combos: set[frozenset[str]] = set()
    selected: list[dict[str, Any]] = []
    stratum_examples: dict[str, list[str]] = defaultdict(list)

    def pick(stratum: dict[str, Any], *, allow_reuse: bool) -> int:
        bucket = stratum["dataset_bucket"]
        qtype = stratum["question_type"]
        need_gap = stratum["gap_constrained"]
        need_ps = stratum["param_similar"]
        want = stratum["quota"] - stratum["filled"]
        if want <= 0:
            return 0
        pool = [
            e
            for e in bucket_entries.get(bucket, [])
            if e["question_type"] == qtype
            and (e["gap_ok"] or not need_gap)
            and (e["ps_ok"] or not need_ps)
        ]
        # prefer under-used budget tiers; rng already shuffled entries
        pool.sort(key=lambda e: tier_counts[e["budget"]])
        taken = 0
        for entry in pool:
            if taken >= want:
                break
            ids = {p.name for p in entry["paths"]}
            combo_key = frozenset(ids)
            if combo_key in picked_combos:
                continue
            if not allow_reuse and ids & seen_candidates:
                continue
            entry["_stratum"] = stratum
            selected.append(entry)
            picked_combos.add(combo_key)
            seen_candidates.update(ids)
            tier_counts[entry["budget"]] += 1
            stratum["filled"] += 1
            if allow_reuse:
                stratum["relaxed_filled"] += 1
            taken += 1
        return taken

    slot_order = sorted(
        plan,
        key=lambda s: (
            DATASET_BUCKETS.index(s["dataset_bucket"]),
            QUESTION_TYPES.index(s["question_type"]),
            not s["gap_constrained"],  # constrained first
            not s["param_similar"],  # capped first
        ),
    )
    for stratum in slot_order:
        pick(stratum, allow_reuse=False)
    if args.allow_reuse_fallback:
        for stratum in slot_order:
            pick(stratum, allow_reuse=True)

    total_filled = sum(s["filled"] for s in plan)
    shortfalls = [
        {**{k: s[k] for k in ("dataset_bucket", "question_type", "gap_constrained", "param_similar")},
         "quota": s["quota"], "filled": s["filled"]}
        for s in plan
        if s["filled"] < s["quota"]
    ]
    if total_filled < args.num_questions:
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "requested": args.num_questions,
            "filled": total_filled,
            "shortfalls": shortfalls,
            "supply": supply,
        }
        write_json(out / "assembly_shortfall.json", report)
        raise SystemExit(
            f"only selected {total_filled}/{args.num_questions} questions; "
            f"shortfall detail written to {out / 'assembly_shortfall.json'}"
        )

    # ---- write question dirs ----
    qdir = out / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    collection: dict[str, Any] = {
        "collection_id": f"v1_llm_{short_hash({'n': args.num_questions, 'seed': args.seed})}",
        "ordered": False,
        "question_paths": [],
        "records": [],
    }
    stats = defaultdict(int)
    gap_values: dict[str, list[float]] = defaultdict(list)
    batch_size_counter: dict[str, int] = defaultdict(int)
    model_choice_counter: dict[str, int] = defaultdict(int)
    winner_model_counter: dict[str, int] = defaultdict(int)
    param_actual_ok = 0

    for idx, entry in enumerate(selected):
        stratum = entry["_stratum"]
        set_path = entry["paths"][0].parent
        dataset_path = set_path.parents[1]
        dataset_spec = read_json(dataset_path / "dataset_spec.json")
        meta = {
            "family": dataset_spec["family"],
            "dataset_id": dataset_spec["dataset_id"],
            "dataset_bucket": stratum["dataset_bucket"],
            "budget_tier": entry["budget"],
            "question_type": stratum["question_type"],
            "gap_constrained": stratum["gap_constrained"],
            "param_similar": stratum["param_similar"],
            "gap": entry["gap"],
            "gap_cap": entry["gap_cap"],
            "win_rate": entry["win_rate"],
            "param_ratio": entry["param_ratio"],
            "batch_sizes": entry["batch_sizes"],
            "trainable_parameter_counts": entry["trainable_parameter_counts"],
            "model_types": entry["model_types"],
            "winner_model_type": entry["winner_model_type"],
        }
        record = build_question_record(
            profile,
            dataset_spec=dataset_spec,
            dataset_path=dataset_path,
            candidate_paths=entry["paths"],
            candidate_set_paths=[set_path],
            rng=rng,
            quality=filters,
            artifact_root=data_root,
            benchmark_metadata=meta,
        )
        qid = record["question_id"]
        qpath = qdir / qid
        qpath.mkdir(parents=True, exist_ok=True)
        record["question_run_id"] = "benchmark_v1_llm"
        record["question_run_path"] = str(qpath.relative_to(out))
        write_json(qpath / "question.json", record)
        if not args.no_prompts:
            write_prompt(qpath, artifact_root=data_root)
        collection["question_paths"].append(str(qpath.relative_to(out)))
        collection["records"].append({"question_id": qid, "order": idx, **meta})
        key = f"{meta['dataset_bucket']}|{meta['question_type']}"
        if len(stratum_examples[key]) < 5:
            stratum_examples[key].append(qid)
        stats[("dataset_bucket", meta["dataset_bucket"])] += 1
        stats[("question_type", meta["question_type"])] += 1
        stats[("budget_tier", meta["budget_tier"])] += 1
        stats[("gap_constrained", meta["gap_constrained"])] += 1
        stats[("param_similar", meta["param_similar"])] += 1
        gap_values[meta["dataset_bucket"]].append(entry["gap"])
        for bs in set(entry["batch_sizes"]):
            batch_size_counter[str(bs)] += 1
        for mt in set(entry["model_types"]):
            model_choice_counter[mt] += 1
        winner_model_counter[entry["winner_model_type"]] += 1
        if entry["param_ratio"] <= args.param_ratio_cap:
            param_actual_ok += 1

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    relaxed_total = sum(s["relaxed_filled"] for s in plan)
    manifest = {
        "name": "v1_llm",
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "created_from": str(data_root),
        "config": {
            "num_questions": len(selected),
            "num_choices": args.num_choices,
            "seed": args.seed,
            "gap_k": args.gap_k,
            "param_ratio_cap": args.param_ratio_cap,
            "max_failed_seeds": args.max_failed_seeds,
            "dataset_buckets": list(DATASET_BUCKETS),
            "budget_tiers": list(BUDGET_TIERS),
            "allow_reuse_fallback": args.allow_reuse_fallback,
        },
        "selection_metric_by_family": selection_metric_by_family,
        "generation_notes": {
            "kan_decision": "kept",
            "kan_evidence": (
                "Per-candidate GT wall times on the A100 under 12-way GPU sharing: "
                "kan median 46.7s vs mlp 22.5s (~2x, not pathological); stuck/stale "
                "candidates were all bigram transformer_lm/gru_lm (crashed-run leftovers "
                "plus one wedged bigram transformer job), zero KAN. KAN kept in all pools."
            ),
        },
        "stats": {f"{k[0]}:{k[1]}": v for k, v in sorted(stats.items(), key=lambda kv: str(kv[0]))},
        "relaxed_questions": relaxed_total,
        "collection": collection["collection_id"],
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "collection.json", collection)

    stage_report = {
        "stage": "v1_llm_assembly",
        "total_questions": len(selected),
        "by_dataset_bucket": {b: stats[("dataset_bucket", b)] for b in DATASET_BUCKETS},
        "by_question_type": {t: stats[("question_type", t)] for t in QUESTION_TYPES},
        "by_budget_tier": {str(t): stats[("budget_tier", t)] for t in BUDGET_TIERS},
        "gap_split": {
            "constrained": stats[("gap_constrained", True)],
            "unconstrained": stats[("gap_constrained", False)],
        },
        "param_split": {
            "capped_slot": stats[("param_similar", True)],
            "unrestricted_slot": stats[("param_similar", False)],
            "actual_ratio_within_cap": param_actual_ok,
        },
        "gap_stats_by_bucket": {
            b: {"p10": pct(v, 0.1), "median": pct(v, 0.5), "p90": pct(v, 0.9)}
            for b, v in gap_values.items()
        },
        "model_type_choice_counts": dict(sorted(model_choice_counter.items())),
        "winner_model_type_counts": dict(sorted(winner_model_counter.items())),
        "batch_size_choice_counts": dict(sorted(batch_size_counter.items())),
        "gt_pool": {
            fam: dict(counts) for fam, counts in sorted(gt_stats.items())
        },
        "plan_vs_achieved": [
            {
                **{k: s[k] for k in ("dataset_bucket", "question_type", "gap_constrained", "param_similar")},
                "quota": s["quota"],
                "filled": s["filled"],
                "relaxed_filled": s["relaxed_filled"],
            }
            for s in slot_order
        ],
        "shortfalls": shortfalls,
        "relaxed_questions": relaxed_total,
        "combo_supply_by_bucket": supply,
        "examples_by_stratum": {k: v for k, v in sorted(stratum_examples.items())},
    }
    write_json(out / "stage_report.json", stage_report)
    print(json.dumps({"questions": len(selected), "manifest": str(out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
