"""Summarize the full-30 protocol correction from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE_A_ROOT = (
    REPO_ROOT / "data/meta_model_studies/setting_to_loss_60q_id_v1/experiments"
)
DEFAULT_INITIAL_ROOT = (
    REPO_ROOT / "data/meta_model_studies/wide_v2_full30_initial_matrix"
)
DEFAULT_ALIGNED_ID_ROOT = (
    REPO_ROOT / "data/meta_model_studies/wide_v2_full30_aligned_et_xgb"
)
DEFAULT_FAMILY_LOGO_ROOT = (
    REPO_ROOT
    / "data/meta_model_studies/wide_v2_full30_family_logo_recovery/unaware_with_params"
)
DEFAULT_FAMILY_LOGO_NO_PARAMS_ROOT = (
    REPO_ROOT
    / "data/meta_model_studies/wide_v2_full30_family_logo_recovery/unaware_no_params"
)
DEFAULT_SNAPSHOT = REPO_ROOT / "artifacts/wide_v2_full30_gt_snapshot.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/wide_v2_full30_protocol_recovery.json"
ANCHOR_ENVIRONMENTS = (
    "bigram_bg_0021c1_b5120_bs64",
    "multi_mvar_c59a30_b5120_bs32",
    "uni_sym_62678b_b2048_bs32",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _method(aggregate: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in aggregate["methods"] if item["method"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} result")
    return matches[0]


def _metrics(item: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in ("all", "benchmark_eligible"):
        payload = item["test"][view]
        within = payload["within_environment"]
        output[view] = {
            "n_rows": payload["n"],
            "macro_three_choice_accuracy": within["macro"][
                "three_choice_accuracy"
            ],
            "micro_three_choice_accuracy": within["three_choice"]["accuracy"],
        }
    return output


def _fixed_methods(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {item["method"]: _metrics(item) for item in aggregate["methods"]}


def _phase_a_fixed(phase_a_root: Path, method_name: str) -> dict[str, Any]:
    per_environment: dict[str, float] = {}
    for path in sorted(phase_a_root.glob("*/leaderboard.json")):
        leaderboard = _load(path)
        item = next(
            item for item in leaderboard["methods"] if item["method"] == method_name
        )
        per_environment[str(leaderboard["experiment_id"])] = item["validation"][
            "all"
        ]["three_choice"]["accuracy"]
    if len(per_environment) != 3:
        raise ValueError("Phase-A summary needs exactly three environments")
    return {
        "method": method_name,
        "per_environment": per_environment,
        "macro_three_choice_accuracy": fmean(per_environment.values()),
    }


def _cv_selected_macro(task_root: Path) -> dict[str, Any]:
    selected: dict[str, str] = {}
    values: dict[str, list[float]] = {"all": [], "benchmark_eligible": []}
    for path in sorted(task_root.glob("*/leaderboard.json")):
        leaderboard = _load(path)
        champion = str(leaderboard["cv_champion"])
        selected[str(leaderboard["task_id"])] = champion
        item = next(
            item for item in leaderboard["methods"] if item["method"] == champion
        )
        for view in values:
            per_environment = item["test"][view]["per_environment"]
            values[view].extend(
                float(metrics["three_choice"]["all"]["accuracy"])
                for metrics in per_environment.values()
            )
    return {
        "selected_methods": selected,
        "all_macro_three_choice_accuracy": fmean(values["all"]),
        "benchmark_eligible_macro_three_choice_accuracy": fmean(
            values["benchmark_eligible"]
        ),
        "selection_rule": "minimum training-only inner-CV log RMSE per task",
    }


def _wide_anchor_bridge(initial_environment: dict[str, Any]) -> dict[str, Any]:
    item = _method(initial_environment, "extra_trees")
    per_environment = item["test"]["all"]["per_environment"]
    anchor_values = {
        environment: per_environment[environment]["three_choice"]["all"]["accuracy"]
        for environment in ANCHOR_ENVIRONMENTS
    }
    return {
        "method": "extra_trees",
        "environments": anchor_values,
        "macro_three_choice_accuracy": fmean(anchor_values.values()),
    }


def summarize(
    *,
    phase_a_root: Path,
    initial_root: Path,
    aligned_id_root: Path,
    family_logo_root: Path,
    family_logo_no_params_root: Path,
    snapshot: Path,
) -> dict[str, Any]:
    initial_environment_path = (
        initial_root
        / "environment_id__unaware__with_params/id/environment/aggregate.json"
    )
    initial_family_path = (
        initial_root
        / "family_pooled_id__unaware__with_params/id/family/aggregate.json"
    )
    aligned_path = aligned_id_root / "id/family/aggregate.json"
    family_logo_path = family_logo_root / "ood/family_logo/aggregate.json"
    family_logo_locked_path = (
        family_logo_root / "ood/family_logo/locked_validation_aggregate.json"
    )
    required = (
        snapshot,
        initial_environment_path,
        initial_family_path,
        aligned_path,
        family_logo_path,
        family_logo_locked_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing recovery inputs: " + ", ".join(missing))

    initial_environment = _load(initial_environment_path)
    initial_family = _load(initial_family_path)
    aligned = _load(aligned_path)
    family_logo = _load(family_logo_path)
    family_logo_locked = _load(family_logo_locked_path)
    result: dict[str, Any] = {
        "schema_version": "wide_v2_full30_protocol_recovery_v1",
        "status": "complete",
        "correction": {
            "retracted_claim": (
                "family_pooled_id was leave-one-family-out and the 84.64% to "
                "68.08% gap was cross-family generalization loss"
            ),
            "fact": (
                "family_pooled_id trains and tests inside each family; a separate "
                "leave_one_family_out run is required"
            ),
        },
        "snapshot": {
            "path": str(snapshot.resolve()),
            "sha256": _sha256(snapshot),
            "counts": _load(snapshot)["counts"],
        },
        "bridge": {
            "phase_a_fixed_extra_trees": _phase_a_fixed(
                phase_a_root, "extra_trees"
            ),
            "wide_v2_same_three_environments_fixed_extra_trees": (
                _wide_anchor_bridge(initial_environment)
            ),
            "full30_per_environment_fixed_extra_trees": _metrics(
                _method(initial_environment, "extra_trees")
            ),
            "full30_family_pooled_id_fixed_extra_trees": _metrics(
                _method(initial_family, "extra_trees")
            ),
        },
        "aligned_family_pooled_id": {
            "protocol": aligned["protocol"],
            "fixed_methods": _fixed_methods(aligned),
            "cv_selected_per_family": _cv_selected_macro(
                aligned_id_root / "id/family"
            ),
            "aggregate_path": str(aligned_path.resolve()),
        },
        "true_family_logo": {
            "protocol": family_logo["protocol"],
            "primary_predeclared_method": "extra_trees",
            "all_held_family_rows": _fixed_methods(family_logo),
            "locked_validation_rows": _fixed_methods(family_logo_locked),
            "cv_selected_per_held_family_all_rows": _cv_selected_macro(
                family_logo_root / "ood/family_logo"
            ),
            "aggregate_path": str(family_logo_path.resolve()),
            "locked_validation_aggregate_path": str(
                family_logo_locked_path.resolve()
            ),
        },
    }
    no_params_path = (
        family_logo_no_params_root / "ood/family_logo/aggregate.json"
    )
    no_params_locked_path = (
        family_logo_no_params_root
        / "ood/family_logo/locked_validation_aggregate.json"
    )
    if no_params_path.is_file() and no_params_locked_path.is_file():
        result["true_family_logo_no_parameter_count"] = {
            "all_held_family_rows": _fixed_methods(_load(no_params_path)),
            "locked_validation_rows": _fixed_methods(_load(no_params_locked_path)),
            "aggregate_path": str(no_params_path.resolve()),
            "locked_validation_aggregate_path": str(no_params_locked_path.resolve()),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    parser.add_argument("--initial-root", type=Path, default=DEFAULT_INITIAL_ROOT)
    parser.add_argument("--aligned-id-root", type=Path, default=DEFAULT_ALIGNED_ID_ROOT)
    parser.add_argument("--family-logo-root", type=Path, default=DEFAULT_FAMILY_LOGO_ROOT)
    parser.add_argument(
        "--family-logo-no-params-root",
        type=Path,
        default=DEFAULT_FAMILY_LOGO_NO_PARAMS_ROOT,
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize(
        phase_a_root=args.phase_a_root,
        initial_root=args.initial_root,
        aligned_id_root=args.aligned_id_root,
        family_logo_root=args.family_logo_root,
        family_logo_no_params_root=args.family_logo_no_params_root,
        snapshot=args.snapshot,
    )
    _write_json(args.output, summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
