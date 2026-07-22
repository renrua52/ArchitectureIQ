"""Reusable deterministic audits for question expansion artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import (
    infer_axes,
    infer_question_type,
    spec_axis_json,
    choices_compatible,
)
from architecture_iq.candidates.sets import list_candidates_in_set
from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile, validate_execution_device
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.significance.validator import validate_significance
from architecture_iq.util import read_json


PRIVATE_PROMPT_MARKERS = (
    "correct_letter",
    "correct_candidate_id",
    "choice_mean_metrics",
    "mean_test_",
    "final_test_",
    "seed_results",
    "results/summary.json",
    "results\\summary.json",
    "significance",
)

HISTORICAL_PROVENANCE_FILENAME = "historical_provenance.json"
HISTORICAL_PROVENANCE_SCHEMA = "historical_candidate_set_provenance_v1"


def _load_historical_provenance(set_path: Path, profile: Profile) -> dict[str, Any] | None:
    path = set_path / HISTORICAL_PROVENANCE_FILENAME
    if not path.is_file():
        return None
    payload = read_json(path)
    verification = payload.get("verification", {})
    if (
        payload.get("schema_version") != HISTORICAL_PROVENANCE_SCHEMA
        or payload.get("status") != "reconstructed_verified"
        or payload.get("historical_profile") != profile.name
        or not payload.get("historical_profile_hash")
        or not payload.get("source_git_commit")
        or not verification.get("single_gt_commit")
        or not verification.get("historical_profile_found")
        or not verification.get("all_candidate_directory_ids_match")
    ):
        return None
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _check(checks: dict[str, dict[str, Any]], name: str, ok: bool, detail: str = "") -> None:
    checks[name] = {"passed": bool(ok), "detail": detail}


def _status(checks: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    failed_names = [name for name, item in checks.items() if not item["passed"]]
    reasons = [f"{name}: {checks[name]['detail']}" for name in failed_names]
    if failed_names and set(failed_names) <= {"profile_hash"}:
        return "review", reasons
    return ("pass" if not reasons else "fail", reasons)


def _canonical_spec(spec: dict[str, Any]) -> str:
    payload = {key: value for key, value in spec.items() if key not in {"candidate_id", "files"}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _candidate_audit(
    candidate_path: Path,
    dataset_spec: dict[str, Any],
    set_manifest: dict[str, Any],
    profile: Profile,
    historical_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    spec_path = candidate_path / "candidate_spec.json"
    summary_path = candidate_path / "results" / "summary.json"
    spec: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    try:
        spec = read_json(spec_path)
    except (OSError, ValueError, TypeError) as exc:
        _check(checks, "candidate_spec", False, str(exc))
        status, reasons = _status(checks)
        return {"candidate_path": str(candidate_path), "candidate_id": candidate_path.name, "status": status, "checks": checks, "reasons": reasons}

    candidate_id = str(spec.get("candidate_id", candidate_path.name))
    _check(checks, "candidate_id", candidate_id == candidate_path.name, "directory name must match candidate_id")
    _check(checks, "dataset_family", spec.get("dataset_id") == dataset_spec.get("dataset_id") and spec.get("family") == dataset_spec.get("family"), "candidate must belong to audited dataset instance")

    provenance_profile = spec.get("profile", set_manifest.get("profile"))
    provenance_hash = spec.get("profile_hash", set_manifest.get("profile_hash"))
    provenance_mode = "current_profile"
    expected_hash = profile.profile_hash
    if historical_provenance is not None:
        provenance_profile = provenance_profile or historical_provenance["historical_profile"]
        provenance_hash = provenance_hash or historical_provenance["historical_profile_hash"]
        expected_hash = str(historical_provenance["historical_profile_hash"])
        provenance_mode = "historical_reconstructed"
    _check(checks, "profile", provenance_profile == profile.name, f"expected {profile.name!r}, got {provenance_profile!r}")
    _check(checks, "profile_hash", provenance_hash == expected_hash, "candidate/set provenance must match the selected or reconstructed historical profile")

    budget = spec.get("budget", {})
    steps, batch, total = budget.get("training_steps"), budget.get("batch_size"), budget.get("total_samples_seen")
    _check(checks, "budget_arithmetic", all(isinstance(value, int) and value > 0 for value in (steps, batch, total)) and steps * batch == total, "training_steps × batch_size must equal total_samples_seen")
    family_default = profile.family_training_defaults(str(dataset_spec["family"]))
    allowed_budgets = set(profile.budget_values)
    if family_default:
        allowed_budgets.add(family_default["total_samples_seen"])
    _check(
        checks,
        "budget_pool",
        total in allowed_budgets,
        f"budget {total!r} must be in profile or an explicit family training default",
    )
    _check(checks, "batch_size_pool", batch in profile.optimizer_grids.get("batch_size", []), f"batch_size {batch!r} must be in profile")

    ensure_registries()
    model = spec.get("model", {})
    model_type = model.get("type")
    try:
        family = get_dataset_family(str(dataset_spec["family"]))
        model_family = get_model_type(str(model_type))
        model_family.validate(model)
        model_ok = model_type in profile.model_types_for_family(str(dataset_spec["family"]), family.compatible_model_types())
        _check(checks, "model", model_ok, "model must be registered, valid, profile-allowed, and family-compatible")
    except (KeyError, TypeError, ValueError) as exc:
        _check(checks, "model", False, str(exc))

    optimizer = spec.get("optimizer", {})
    _check(checks, "optimizer", optimizer.get("type") in profile.pools.get("optimizers", []), "optimizer type must be allowed by profile")
    loss = spec.get("loss", {})
    _check(checks, "loss", loss.get("loss_id") in profile.pools.get("losses", {}).get(dataset_spec.get("family"), []), "loss must be allowed for dataset family")

    execution = spec.get("execution", {}).get("device")

    try:
        summary = read_json(summary_path)
        _check(checks, "summary_exists", True)
    except (OSError, ValueError, TypeError) as exc:
        _check(checks, "summary_exists", False, str(exc))
        status, reasons = _status(checks)
        return {"candidate_path": str(candidate_path), "candidate_id": candidate_id, "status": status, "provenance_mode": provenance_mode, "checks": checks, "reasons": reasons, "normalized_spec": _canonical_spec(spec)}

    metric = str(dataset_spec["selection_metric"])
    _check(checks, "summary_candidate_id", summary.get("candidate_id") == candidate_id, "summary candidate_id must match spec")
    _check(checks, "summary_metric", summary.get("selection_metric") == metric, "summary selection_metric must match dataset")
    _check(checks, "summary_excluded", not bool(summary.get("excluded")), "excluded candidates cannot be reused")
    _check(checks, "summary_seed_count", summary.get("n_seeds") == profile.n_seeds and len(summary.get("seed_results", [])) == profile.n_seeds, "summary seed count must match profile")
    _check(checks, "summary_base_seed", summary.get("base_seed") == profile.base_seed, "summary base_seed must match profile")
    seed_results = summary.get("seed_results", [])
    expected_seeds = list(range(profile.base_seed, profile.base_seed + profile.n_seeds))
    _check(checks, "seed_alignment", [item.get("seed") for item in seed_results] == expected_seeds, "seed ids must be ordered and aligned")
    failed = sum(bool(item.get("failed")) for item in seed_results)
    _check(checks, "failed_seeds", summary.get("failed_seeds") == failed and failed <= int(profile.ground_truth["max_failed_seeds"]), "failed seed count must be truthful and within profile limit")
    metric_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    final_key = f"final_{metric}"
    finite_seed_metrics = all(bool(item.get("failed")) or _finite(item.get(final_key)) for item in seed_results)
    _check(checks, "finite_metrics", _finite(summary.get(metric_key)) and _finite(summary.get(std_key)) and finite_seed_metrics, "mean, std, and non-failed final metrics must be finite")
    _check(checks, "curves", (candidate_path / "results" / "curves.npz").is_file(), "results/curves.npz is required for reusable GT")
    summary_device = summary.get("environment", {}).get("device")
    resolved_device = execution if execution is not None else summary_device
    try:
        validate_execution_device(str(resolved_device))
        _check(
            checks,
            "execution_device",
            True,
            "from candidate spec" if execution is not None else "legacy candidate: derived from summary environment",
        )
        _check(
            checks,
            "device_provenance",
            execution is None or execution == summary_device,
            "candidate execution.device must match summary environment device when both are present",
        )
    except ValueError as exc:
        _check(checks, "execution_device", False, str(exc))
        _check(checks, "device_provenance", False, "no valid execution device provenance")

    status, reasons = _status(checks)
    return {"candidate_path": str(candidate_path), "candidate_id": candidate_id, "status": status, "provenance_mode": provenance_mode, "checks": checks, "reasons": reasons, "normalized_spec": _canonical_spec(spec)}


def audit_question_inputs(dataset_path: Path, candidate_set_paths: list[Path], profile: Profile) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    set_reports: list[dict[str, Any]] = []
    normalized: dict[str, list[str]] = {}
    for set_path in candidate_set_paths:
        set_path = set_path.resolve()
        manifest = read_json(set_path / "set.json")
        historical_provenance = _load_historical_provenance(set_path, profile)
        candidates = [_candidate_audit(path, dataset_spec, manifest, profile, historical_provenance) for path in list_candidates_in_set(set_path)]
        for item in candidates:
            normalized.setdefault(item["normalized_spec"], []).append(item["candidate_id"])
        checks: dict[str, dict[str, Any]] = {}
        _check(checks, "set_dataset_family", manifest.get("dataset_id") == dataset_spec.get("dataset_id") and manifest.get("family") == dataset_spec.get("family"), "set must belong to audited dataset")
        effective_profile_hash = (
            historical_provenance["historical_profile_hash"]
            if historical_provenance is not None
            else profile.profile_hash
        )
        effective_manifest_hash = manifest.get("profile_hash") or effective_profile_hash
        _check(
            checks,
            "set_profile",
            manifest.get("profile") == profile.name and effective_manifest_hash == effective_profile_hash,
            "set profile provenance must match selected or reconstructed historical profile",
        )
        _check(
            checks,
            "historical_provenance",
            historical_provenance is not None or manifest.get("profile_hash") is not None,
            "legacy sets without profile_hash require a verified historical_provenance.json sidecar",
        )
        _check(checks, "set_count", manifest.get("count") == len(candidates), "set manifest count must match candidate directories")
        specs = [read_json(Path(item["candidate_path"]) / "candidate_spec.json") for item in candidates]
        for axis in ("model", "optimizer", "loss"):
            values = {spec_axis_json(spec, axis) for spec in specs}
            declared_varying = axis in manifest.get("varying_axes", [])
            _check(checks, f"axis_{axis}", (len(values) > 1) == declared_varying, "declared varying_axes must match member configurations")
        set_status, set_reasons = _status(checks)
        if any(item["status"] == "fail" for item in candidates):
            set_status = "fail"
            set_reasons.append("one or more candidate audits failed")
        elif any(item["status"] == "review" for item in candidates) and set_status == "pass":
            set_status = "review"
            set_reasons.append("one or more candidates require provenance review")
        set_reports.append({"candidate_set": str(set_path), "status": set_status, "checks": checks, "reasons": set_reasons, "historical_provenance": historical_provenance, "candidates": candidates})

    collisions = sorted(sorted(ids) for ids in normalized.values() if len(ids) > 1)
    if collisions:
        collided = {candidate_id for ids in collisions for candidate_id in ids}
        for report in set_reports:
            for candidate in report["candidates"]:
                if candidate["candidate_id"] in collided:
                    candidate["status"] = "fail"
                    candidate["reasons"].append("normalized candidate spec collides with another candidate_id")
            if any(candidate["status"] == "fail" for candidate in report["candidates"]):
                report["status"] = "fail"
    all_candidates = [candidate for report in set_reports for candidate in report["candidates"]]
    return {
        "schema_version": "question_input_audit_v1",
        "gate": 1,
        "gate_2": 2,
        "dataset_path": str(dataset_path),
        "dataset_id": dataset_spec["dataset_id"],
        "family": dataset_spec["family"],
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "candidate_sets": set_reports,
        "summary": {
            "candidates": len(all_candidates),
            "pass": sum(item["status"] == "pass" for item in all_candidates),
            "review": sum(item["status"] == "review" for item in all_candidates),
            "fail": sum(item["status"] == "fail" for item in all_candidates),
            "reusable_candidate_ids": [item["candidate_id"] for item in all_candidates if item["status"] == "pass"],
            "review_candidate_ids": [item["candidate_id"] for item in all_candidates if item["status"] == "review"],
            "failed_candidate_ids": [item["candidate_id"] for item in all_candidates if item["status"] == "fail"],
            "normalized_spec_collisions": collisions,
        },
        "valid": all(report["status"] == "pass" for report in set_reports),
    }


def audit_question_run(run_path: Path, profile: Profile, *, data_root: Path = DATA_DIR) -> dict[str, Any]:
    run_path = run_path.resolve()
    data_root = data_root.resolve()
    manifest = read_json(run_path / "run.json")
    question_dirs = sorted(path for path in run_path.iterdir() if (path / "question.json").is_file())
    run_checks: dict[str, dict[str, Any]] = {}
    _check(run_checks, "manifest_count", manifest.get("num_questions") == len(question_dirs) == len(manifest.get("question_ids", [])), "run manifest count and question_ids must match directories")
    _check(run_checks, "manifest_profile", manifest.get("profile") == profile.name and manifest.get("profile_hash") == profile.profile_hash, "run profile provenance must match selected profile")
    reports: list[dict[str, Any]] = []
    all_candidate_ids: list[str] = []
    for question_dir in question_dirs:
        question = read_json(question_dir / "question.json")
        checks: dict[str, dict[str, Any]] = {}
        choices = question.get("choices", [])
        _check(checks, "question_id", question.get("question_id") == question_dir.name, "question_id must match directory")
        _check(checks, "choice_count", len(choices) >= 2 and len(choices) == question.get("num_choices"), "question must declare at least two choices")
        paths: list[Path] = []
        specs: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for choice in choices:
            path = data_root / str(choice.get("candidate_path", ""))
            paths.append(path)
            try:
                spec = read_json(path / "candidate_spec.json")
                summary = read_json(path / "results" / "summary.json")
                specs.append(spec)
                summaries.append(summary)
                _check(checks, f"candidate_{choice.get('letter')}", spec.get("candidate_id") == choice.get("candidate_id") and spec.get("dataset_id") == question.get("dataset_id") and spec.get("family") == question.get("family"), "choice provenance must match source spec")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                _check(checks, f"candidate_{choice.get('letter')}", False, str(exc))
        candidate_ids = [str(choice.get("candidate_id")) for choice in choices]
        all_candidate_ids.extend(candidate_ids)
        _check(checks, "choice_ids", len(candidate_ids) == len(set(candidate_ids)), "candidate ids must be unique within question")
        if len(specs) == len(choices):
            _check(checks, "compatibility", choices_compatible(specs), "choices must be compatible")
            invariant, varying = infer_axes(specs)
            _check(checks, "axes", question.get("invariant_axes") == invariant and question.get("varying_axes") == varying and question.get("type") == infer_question_type(specs), "recorded type and axes must match specs")
            metric = str(question.get("evaluation", {}).get("selection_metric", ""))
            try:
                sig = validate_significance(summaries, profile, metric=metric)
                recorded = question.get("significance", {})
                _check(checks, "significance", sig.passed and bool(recorded.get("passed")) and recorded.get("metric") == metric and math.isclose(float(recorded.get("gap", -1)), sig.gap, rel_tol=1e-12) and math.isclose(float(recorded.get("win_rate", -1)), sig.win_rate, rel_tol=1e-12), sig.reason or "recorded significance must match independent recomputation")
                winner_id = specs[sig.winner_index]["candidate_id"] if sig.winner_index >= 0 else None
                correct = next((choice["candidate_id"] for choice in choices if choice.get("letter") == question.get("correct_letter")), None)
                _check(checks, "winner", winner_id == correct, "correct_letter must point to independently recomputed winner")
            except (KeyError, TypeError, ValueError) as exc:
                _check(checks, "significance", False, str(exc))
            totals = {spec["budget"]["total_samples_seen"] for spec in specs}
            budget = question.get("budget", {})
            expected_budget: dict[str, Any] = {"total_samples_seen": next(iter(totals))} if len(totals) == 1 else {"total_samples_seen": sorted(totals), "mixed": True}
            _check(checks, "budget", budget == expected_budget, "question budget field must match choice specs")
        prompt_path = question_dir / question.get("prompt", {}).get("rendered_path", "prompt.txt")
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
            marker = next((item for item in PRIVATE_PROMPT_MARKERS if item in prompt.lower()), None)
            _check(checks, "public_prompt", marker is None, f"private prompt marker: {marker!r}")

        except OSError as exc:
            _check(checks, "public_prompt", False, str(exc))
        status, reasons = _status(checks)
        reports.append({"question_id": question.get("question_id", question_dir.name), "question_dir": str(question_dir), "status": status, "checks": checks, "reasons": reasons})
    _check(run_checks, "candidate_disjoint", len(all_candidate_ids) == len(set(all_candidate_ids)), "candidate_id may not be reused within run")
    run_status, run_reasons = _status(run_checks)
    valid = run_status == "pass" and all(item["status"] == "pass" for item in reports)
    return {"schema_version": "question_run_audit_v1", "gate": 3, "gate_4": 4, "question_run": str(run_path), "profile": profile.name, "profile_hash": profile.profile_hash, "run_checks": run_checks, "run_reasons": run_reasons, "questions": reports, "summary": {"questions": len(reports), "pass": sum(item["status"] == "pass" for item in reports), "fail": sum(item["status"] != "pass" for item in reports), "unique_candidates": len(set(all_candidate_ids))}, "valid": valid}


def markdown_report(title: str, report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [f"# {title}", "", f"- valid: `{report['valid']}`", f"- profile: `{report['profile']}`", f"- pass/fail: {summary['pass']}/{summary['fail']}", "", "## Failed items", ""]
    failures: list[tuple[str, list[str]]] = []
    if "candidate_sets" in report:
        for candidate_set in report["candidate_sets"]:
            for candidate in candidate_set["candidates"]:
                if candidate["status"] != "pass":
                    failures.append((candidate["candidate_id"], candidate["reasons"]))
    else:
        for question in report["questions"]:
            if question["status"] != "pass":
                failures.append((question["question_id"], question["reasons"]))
    if not failures:
        lines.append("None.")
    else:
        for item, reasons in failures:
            lines.append(f"- `{item}`: {'; '.join(reasons)}")
    return "\n".join(lines) + "\n"
