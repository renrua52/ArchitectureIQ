#!/usr/bin/env python3
"""Summarize trainable-parameter ratios for a frozen question collection.

The report reads only existing question and candidate artifacts.  It is meant to
inform thresholds in a *new* profile, never to rewrite historical specs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _relative_artifact_path(value: str) -> Path:
    """Normalize manifests produced on either Windows or POSIX."""
    return Path(value.replace("\\", "/"))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    ratios = [item["parameter_ratio"] for item in items]
    log2_ratios = [item["log2_parameter_ratio"] for item in items]
    return {
        "questions": len(items),
        "ratio": {
            "min": min(ratios),
            "p10": _percentile(ratios, 0.10),
            "p25": _percentile(ratios, 0.25),
            "median": median(ratios),
            "p75": _percentile(ratios, 0.75),
            "p90": _percentile(ratios, 0.90),
            "max": max(ratios),
            "mean": mean(ratios),
        },
        "log2_ratio": {
            "median": median(log2_ratios),
            "p90": _percentile(log2_ratios, 0.90),
            "max": max(log2_ratios),
        },
        "within_1_25x": sum(ratio <= 1.25 for ratio in ratios),
        "within_1_5x": sum(ratio <= 1.5 for ratio in ratios),
        "within_2x": sum(ratio <= 2.0 for ratio in ratios),
        "over_4x": sum(ratio > 4.0 for ratio in ratios),
        "over_10x": sum(ratio > 10.0 for ratio in ratios),
    }


def _markdown_table(groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | n | min | p25 | median | p75 | p90 | max | <=2x | >4x | >10x |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in groups.items():
        ratio = summary["ratio"]
        count = summary["questions"]
        lines.append(
            "| {name} | {n} | {min_} | {p25} | {median_} | {p75} | {p90} | {max_} | "
            "{within}/{n} | {over4}/{n} | {over10}/{n} |".format(
                name=name,
                n=count,
                min_=_fmt(ratio["min"]),
                p25=_fmt(ratio["p25"]),
                median_=_fmt(ratio["median"]),
                p75=_fmt(ratio["p75"]),
                p90=_fmt(ratio["p90"]),
                max_=_fmt(ratio["max"]),
                within=summary["within_2x"],
                over4=summary["over_4x"],
                over10=summary["over_10x"],
            )
        )
    return lines


def build_report(repo_root: Path, collection_path: Path) -> tuple[dict[str, Any], str]:
    collection = _json(collection_path)
    data_root = repo_root / "data"
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for record in collection["records"]:
        question_path = data_root / _relative_artifact_path(record["question_path"])
        question = _json(question_path / "question.json")
        counts: list[int] = []
        for choice in question["choices"]:
            spec_path = data_root / _relative_artifact_path(choice["candidate_path"]) / "candidate_spec.json"
            spec = _json(spec_path)
            count = spec.get("trainable_parameter_count")
            if not isinstance(count, int) or count <= 0:
                missing.append({"question_id": record["question_id"], "candidate_id": choice["candidate_id"]})
                continue
            counts.append(count)
        if len(counts) != len(question["choices"]):
            continue
        smallest, largest = min(counts), max(counts)
        ratio = largest / smallest
        rows.append(
            {
                "order": record["order"],
                "question_id": record["question_id"],
                "track": record.get("track", "unknown"),
                "profile": record.get("profile", question.get("profile", "unknown")),
                "family": record.get("family", question["family"]),
                "question_type": record.get("question_type", question["type"]),
                "candidate_parameter_counts": counts,
                "parameter_ratio": ratio,
                "log2_parameter_ratio": math.log2(ratio),
                "significance_gap": question.get("significance", {}).get("gap"),
            }
        )

    if missing:
        raise ValueError(f"{len(missing)} choices are missing usable parameter counts: {missing}")
    if not rows:
        raise ValueError("collection contains no analyzable questions")

    groupings = {
        "by_track_profile": lambda row: f"{row['track']} ({row['profile']})",
        "by_family_question_type": lambda row: f"{row['family']} / {row['question_type']}",
        "by_track": lambda row: row["track"],
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for label, key_fn in groupings.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        summaries[label] = {
            name: _summarize(bucket)
            for name, bucket in sorted(buckets.items(), key=lambda item: item[0])
        }

    payload = {
        "schema_version": "parameter_ratio_distribution_v1",
        "collection": str(collection_path.relative_to(repo_root)).replace("\\", "/"),
        "collection_id": collection.get("collection_id"),
        "question_count": len(rows),
        "missing_parameter_counts": missing,
        "questions": rows,
        "summaries": {"overall": _summarize(rows), **summaries},
    }

    overall = payload["summaries"]["overall"]
    md = [
        "# Parameter-ratio distribution report",
        "",
        "## Scope",
        "",
        f"- Collection: `{payload['collection']}` (`{payload['collection_id']}`)",
        f"- Analyzed question pairs: {len(rows)}; missing usable parameter counts: {len(missing)}.",
        "- Ratio is `max(trainable_parameter_count) / min(trainable_parameter_count)` within each question.",
        "- This is descriptive evidence for new profiles only. It does not modify frozen v1/v2/v2.1/v2.2 artifacts.",
        "",
        "## Overall",
        "",
        f"- Range: {_fmt(overall['ratio']['min'])}x–{_fmt(overall['ratio']['max'])}x; median {_fmt(overall['ratio']['median'])}x; p90 {_fmt(overall['ratio']['p90'])}x.",
        f"- {overall['within_2x']}/{overall['questions']} pairs are within 2x; {overall['over_4x']}/{overall['questions']} exceed 4x; {overall['over_10x']}/{overall['questions']} exceed 10x.",
        "",
        "## By track and profile",
        "",
        *_markdown_table(payload["summaries"]["by_track_profile"]),
        "",
        "## By family and question type",
        "",
        *_markdown_table(payload["summaries"]["by_family_question_type"]),
        "",
        "## Largest ratios (manual-review candidates)",
        "",
        "| Question | Track | Family / type | Counts | Ratio | Gap |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: value["parameter_ratio"], reverse=True)[:10]:
        gap = row["significance_gap"]
        md.append(
            f"| `{row['question_id']}` | {row['track']} | {row['family']} / {row['question_type']} | "
            f"{row['candidate_parameter_counts'][0]} vs {row['candidate_parameter_counts'][1]} | "
            f"{_fmt(row['parameter_ratio'])}x | {gap if gap is not None else '—'} |"
        )
    md.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A parameter ratio is a capacity-screening signal, not a universal fairness or difficulty metric. "
            "New easy/hard selection should primarily use the family × question-type gap distribution and GT stability; "
            "ratio thresholds should be profile/track-specific and may allow explicitly labeled diagnostic exceptions.",
            "",
        ]
    )
    return payload, "\n".join(md)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("outputs/demo_release_integration/demo_release_collection_v2.json"),
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    collection = args.collection if args.collection.is_absolute() else repo_root / args.collection
    payload, markdown = build_report(repo_root, collection.resolve())
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.json_output} and {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
