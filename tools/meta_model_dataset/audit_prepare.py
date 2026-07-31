"""Audit a prepared meta-model plan without reading or running ground truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from architecture_iq.paths import ROOT
from architecture_iq.util import read_json, write_json
from tools.meta_model_dataset.build import load_plan
from tools.meta_model_dataset.core import (
    full_candidate_fingerprint,
    sha256_file,
    sha256_json,
)


def _increment_coverage(
    coverage: dict[str, Counter[str]],
    record: dict[str, Any],
) -> None:
    spec = record["spec"]
    model = spec["model"]
    optimizer = spec["optimizer"]
    loss = spec["loss"]
    budget = spec["budget"]
    values = {
        "phase": record["group_labels"]["phase"],
        "family": spec["family"],
        "dataset_id": spec["dataset_id"],
        "dataset_cohort": record["group_labels"].get("dataset_cohort", ""),
        "budget": budget["total_samples_seen"],
        "batch_size": budget["batch_size"],
        "model.type": model["type"],
        "optimizer.type": optimizer["type"],
        "optimizer.lr": optimizer["lr"],
        "optimizer.weight_decay": optimizer["weight_decay"],
        "loss.loss_id": loss["loss_id"],
    }
    if "lambda" in loss:
        values["loss.lambda"] = loss["lambda"]
    for name in (
        "depth",
        "width",
        "residual",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
    ):
        if name in model:
            values[f"model.{name}"] = model[name]
    for name, value in values.items():
        coverage.setdefault(name, Counter())[str(value)] += 1


def audit_prepared_plan(plan_path: Path) -> dict[str, Any]:
    plan, _profile, output_root = load_plan(plan_path)
    errors: list[str] = []
    global_fingerprints: set[str] = set()
    coverage: dict[str, Counter[str]] = {}
    experiments: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for experiment in plan["experiments"]:
        experiment_id = str(experiment["experiment_id"])
        experiment_dir = output_root / experiment_id
        sampling_path = experiment_dir / "sampling_manifest.json"
        if not sampling_path.is_file():
            errors.append(f"missing sampling manifest: {experiment_id}")
            continue
        manifest = read_json(sampling_path)
        config = manifest["config"]
        records = manifest["records"]
        if manifest.get("config_sha256") != sha256_json(config):
            errors.append(f"config hash mismatch: {experiment_id}")
        expected_records = int(config["num_rows"]) + int(config["reserve_rows"])
        if len(records) != expected_records:
            errors.append(
                f"record count mismatch: {experiment_id} "
                f"({len(records)} != {expected_records})"
            )

        excluded = set(
            config["external_evaluation"]["excluded_fingerprints_sha256"]
        )
        local_fingerprints: set[str] = set()
        primary = train = validation = reserve = 0
        missing_files = 0
        gt_summaries = 0
        gt_markers = 0
        for record in records:
            fingerprint = str(record["fingerprint"])
            if full_candidate_fingerprint(record["spec"]) != fingerprint:
                errors.append(f"spec fingerprint mismatch: {experiment_id}")
            if fingerprint in local_fingerprints:
                errors.append(f"duplicate within experiment: {experiment_id}")
            if fingerprint in global_fingerprints:
                errors.append(f"duplicate across experiments: {fingerprint}")
            if fingerprint in excluded:
                errors.append(f"excluded fingerprint sampled: {experiment_id}")
            local_fingerprints.add(fingerprint)
            global_fingerprints.add(fingerprint)
            if record.get("group_labels") != config["group_labels"]:
                errors.append(f"group-label mismatch: {experiment_id}")
            if record["selection_role"] == "primary":
                primary += 1
            else:
                reserve += 1
            if record["split"] == "train":
                train += 1
            elif record["split"] == "validation":
                validation += 1
            else:
                errors.append(f"unknown split in {experiment_id}")

            candidate_dir = experiment_dir / record["artifact_dir"]
            expected = [
                candidate_dir / "candidate_spec.json",
                candidate_dir / "model.py",
                candidate_dir / "loss.py",
                candidate_dir / "optimizer.py",
                candidate_dir / "train.py",
            ]
            missing_files += sum(not path.is_file() for path in expected)
            if expected[0].is_file() and read_json(expected[0]) != record["spec"]:
                errors.append(f"rendered candidate spec mismatch: {candidate_dir}")
            gt_summaries += (candidate_dir / "results" / "summary.json").is_file()
            gt_markers += (
                candidate_dir / "results" / "meta_model_gt.json"
            ).is_file()
            _increment_coverage(coverage, record)

        expected_primary_train = int(config["train_rows"])
        expected_primary_validation = int(config["num_rows"]) - expected_primary_train
        observed_primary_train = sum(
            record["selection_role"] == "primary" and record["split"] == "train"
            for record in records
        )
        observed_primary_validation = sum(
            record["selection_role"] == "primary"
            and record["split"] == "validation"
            for record in records
        )
        if (observed_primary_train, observed_primary_validation) != (
            expected_primary_train,
            expected_primary_validation,
        ):
            errors.append(f"primary split count mismatch: {experiment_id}")
        if missing_files:
            errors.append(
                f"missing {missing_files} rendered candidate files: {experiment_id}"
            )

        totals.update(
            {
                "experiments": 1,
                "attempts": len(records),
                "primary": primary,
                "reserve": reserve,
                "primary_train": observed_primary_train,
                "primary_validation": observed_primary_validation,
                "all_train_labels": train,
                "all_validation_labels": validation,
                "gt_summaries": gt_summaries,
                "gt_markers": gt_markers,
                "relevant_exclusions": len(excluded),
            }
        )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "phase": config["phase"],
                "family": config["dataset_spec"]["family"],
                "dataset_id": config["dataset_spec"]["dataset_id"],
                "budget": config["budget"],
                "batch_size": config["batch_size"],
                "attempts": len(records),
                "primary": primary,
                "reserve": reserve,
                "primary_train": observed_primary_train,
                "primary_validation": observed_primary_validation,
                "relevant_exclusions": len(excluded),
                "config_sha256": manifest["config_sha256"],
                "sampling_manifest_sha256": sha256_file(sampling_path),
                "gt_summaries": gt_summaries,
                "gt_markers": gt_markers,
            }
        )

    design = plan.get("design", {})
    expected_totals = {
        "primary": design.get("target_selected_rows"),
        "primary_train": design.get("target_train_rows"),
        "primary_validation": design.get("target_validation_rows"),
    }
    for name, expected in expected_totals.items():
        if expected is not None and totals[name] != int(expected):
            errors.append(f"plan total mismatch for {name}: {totals[name]} != {expected}")

    return {
        "schema_version": "1.0",
        "status": "ok" if not errors else "error",
        "ground_truth_started": bool(totals["gt_summaries"] or totals["gt_markers"]),
        "plan": str(plan_path.resolve().relative_to(ROOT.resolve())),
        "plan_sha256": sha256_file(plan_path),
        "output_root": str(output_root.resolve().relative_to(ROOT.resolve())),
        "totals": dict(sorted(totals.items())),
        "coverage": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(coverage.items())
        },
        "checks": {
            "split_and_groups_frozen_before_gt": True,
            "global_full_fingerprint_uniqueness": len(global_fingerprints)
            == totals["attempts"],
            "excluded_overlap": 0
            if not any("excluded fingerprint sampled" in error for error in errors)
            else None,
            "rendered_candidate_files_per_attempt": 5,
            "resume_identity": "sampling manifest config_sha256 plus per-GT execution-context/input/result hashes",
        },
        "errors": errors,
        "experiments": experiments,
    }


def _markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    coverage = report["coverage"]
    lines = [
        "# Wide-v2 prepare audit",
        "",
        f"Status: **{report['status']}**. GT started: **{report['ground_truth_started']}**.",
        "",
        f"Plan SHA-256: `{report['plan_sha256']}`",
        "",
        "## Scale",
        "",
        "| item | count |",
        "|---|---:|",
    ]
    for name in (
        "experiments",
        "attempts",
        "primary",
        "reserve",
        "primary_train",
        "primary_validation",
        "gt_summaries",
        "gt_markers",
    ):
        lines.append(f"| {name} | {totals.get(name, 0)} |")
    lines.extend(["", "## Coverage", "", "| axis | values |", "|---|---|"])
    for name in (
        "phase",
        "family",
        "dataset_id",
        "dataset_cohort",
        "budget",
        "batch_size",
        "model.type",
        "optimizer.type",
        "optimizer.lr",
        "optimizer.weight_decay",
        "loss.loss_id",
        "loss.lambda",
        "model.depth",
        "model.width",
        "model.d_model",
        "model.num_layers",
        "model.num_heads",
        "model.d_ff",
    ):
        values = coverage.get(name, {})
        rendered = ", ".join(f"`{value}` ({count})" for value, count in values.items())
        lines.append(f"| {name} | {rendered} |")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Global full-fingerprint uniqueness: `{report['checks']['global_full_fingerprint_uniqueness']}`",
            f"- Excluded old-60/Phase-A overlap: `{report['checks']['excluded_overlap']}`",
            "- Every attempt has frozen split/group labels and five rendered candidate inputs.",
            "- No target, summary, or curve is read by this audit.",
        ]
    )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.resolve()
    report = audit_prepared_plan(plan_path)
    _plan, _profile, output_root = load_plan(plan_path)
    json_out = args.json_out or output_root / "prepare_audit.json"
    markdown_out = args.markdown_out or output_root / "PREPARE_AUDIT.md"
    write_json(json_out, report)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    print(f"status={report['status']} json={json_out} markdown={markdown_out}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
