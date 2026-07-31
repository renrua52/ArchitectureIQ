from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools.meta_model_dataset import export_or_rescue_supplemental as driver
from tools.meta_model_dataset.supplemental_reserve_common import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)


EXACT_PREFIX = "Not enough usable reserve settings to replace "


def _fixture(
    tmp_path: Path,
    experiment_ids: list[str],
) -> dict[str, Any]:
    base_root = tmp_path / "base"
    supplemental_root = tmp_path / "supplemental"
    merged_root = tmp_path / "merged"
    base_plan_path = tmp_path / "base_plan.json"
    supplemental_plan_path = tmp_path / "supplemental_plan.json"
    policy_path = tmp_path / "policy.json"
    contract_path = tmp_path / "freeze.json"
    progress_path = tmp_path / "progress.json"
    experiments = [
        {"experiment_id": experiment_id, "phase": "b2_scale"}
        for experiment_id in experiment_ids
    ]
    base_plan = {
        "schema_version": "2.0",
        "output_root": str(base_root),
        "defaults": {"phase": "b2_scale"},
        "experiments": experiments,
    }
    supplemental_plan = {
        "schema_version": "2.0",
        "output_root": str(supplemental_root),
        "defaults": {"phase": "b2_scale"},
        "experiments": experiments,
    }
    policy = {
        "schema_version": "1.0",
        "contract_id": "test_contract",
        "freeze_contract": str(contract_path),
        "base_output_root": str(base_root),
        "supplemental_output_root": str(supplemental_root),
        "merged_output_root": str(merged_root),
        "activation": {
            "exception_type": "RuntimeError",
            "message_prefix": EXACT_PREFIX,
            "successful_base_exports_immutable": True,
        },
    }
    atomic_write_json(base_plan_path, base_plan)
    atomic_write_json(supplemental_plan_path, supplemental_plan)
    atomic_write_json(policy_path, policy)
    contract = {
        "schema_version": "1.0",
        "contract_id": "test_contract",
        "base_plan": {
            "path": str(base_plan_path),
            "sha256": sha256_file(base_plan_path),
        },
        "supplemental_plan": {
            "path": str(supplemental_plan_path),
            "sha256": sha256_file(supplemental_plan_path),
        },
        "policy_source": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "policy_sha256": sha256_json(policy),
        "experiments": {experiment_id: {} for experiment_id in experiment_ids},
        "timing_snapshot": {
            "before_prepare": {
                "environments": {
                    experiment_id: {
                        "export_exists": False,
                        "opaque_export_file_sha256": {},
                    }
                    for experiment_id in experiment_ids
                }
            }
        },
    }
    contract["content_sha256"] = sha256_json(contract)
    atomic_write_json(contract_path, contract)
    return {
        "base_root": base_root,
        "supplemental_root": supplemental_root,
        "merged_root": merged_root,
        "base_plan": base_plan_path,
        "supplemental_plan": supplemental_plan_path,
        "policy": policy_path,
        "contract": contract_path,
        "progress": progress_path,
    }


def _run(paths: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return driver.run_driver(
        workers=kwargs.pop("workers", 4),
        base_plan_path=paths["base_plan"],
        supplemental_plan_path=paths["supplemental_plan"],
        policy_path=paths["policy"],
        contract_path=paths["contract"],
        progress_path=paths["progress"],
        event_sink=kwargs.pop("event_sink", lambda _event: None),
        **kwargs,
    )


@pytest.mark.parametrize("workers", [0, 5, 99])
def test_driver_hard_limits_workers(tmp_path: Path, workers: int) -> None:
    paths = _fixture(tmp_path, ["env"])

    with pytest.raises(ValueError, match="workers must be between 1 and 4"):
        _run(paths, workers=workers)

    assert not paths["progress"].exists()


def test_dry_run_is_read_only_and_does_not_call_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, ["done", "pending"])
    atomic_write_json(paths["base_root"] / "done" / "manifest.json", {"ok": True})
    monkeypatch.setattr(
        driver.standard_builder,
        "build_from_plan",
        lambda **_kwargs: pytest.fail("dry-run called the builder"),
    )
    events: list[dict[str, Any]] = []

    result = _run(paths, dry_run=True, event_sink=events.append)

    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    assert result["builder_calls_performed"] is False
    assert not paths["progress"].exists()
    assert [item["action"] for item in result["actions"]] == [
        "skip_existing_base_export",
        "attempt_base_export_then_conditionally_run_sidecar_rescue",
    ]
    assert events == [result]


def test_exact_capacity_failure_runs_sidecar_merge_audit_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, ["base_ok", "needs_rescue", "continues"])
    calls: list[tuple[str, str, str, int]] = []

    def fake_build_from_plan(**kwargs: Any) -> list[dict[str, Any]]:
        plan_path = Path(kwargs["plan_path"])
        experiment_id = next(iter(kwargs["requested_experiments"]))
        stage = str(kwargs["stage"])
        workers = int(kwargs["workers"])
        kind = "base" if plan_path == paths["base_plan"].resolve() else "sidecar"
        calls.append((kind, experiment_id, stage, workers))
        if kind == "base" and experiment_id == "needs_rescue":
            raise RuntimeError(f"{EXACT_PREFIX}bad-fingerprint in validation")
        root = paths["base_root"] if kind == "base" else paths["supplemental_root"]
        atomic_write_json(root / experiment_id / "manifest.json", {"ok": True})
        return []

    def fake_merge(*, experiment_id: str, contract_path: Path) -> Path:
        assert contract_path == paths["contract"].resolve()
        output = paths["merged_root"] / experiment_id
        atomic_write_json(output / "manifest.json", {"merged": True})
        return output

    def fake_audit(*, experiment_id: str, contract_path: Path) -> dict[str, Any]:
        assert contract_path == paths["contract"].resolve()
        return {"status": "ok", "experiment_id": experiment_id, "errors": []}

    monkeypatch.setattr(driver.standard_builder, "build_from_plan", fake_build_from_plan)
    monkeypatch.setattr(driver, "merge_experiment", fake_merge)
    monkeypatch.setattr(driver, "audit_merged_experiment", fake_audit)

    result = _run(paths, workers=4)

    assert result["status"] == "complete"
    assert calls == [
        ("base", "base_ok", "export", 4),
        ("base", "needs_rescue", "export", 4),
        ("sidecar", "needs_rescue", "all", 4),
        ("base", "continues", "export", 4),
    ]
    assert result["experiments"]["base_ok"]["status"] == "base_export_complete"
    assert result["experiments"]["needs_rescue"]["status"] == "rescue_complete"
    assert result["experiments"]["continues"]["status"] == "base_export_complete"
    assert (
        paths["merged_root"] / "needs_rescue" / "rescue_audit.json"
    ).is_file()
    stored = read_json(paths["progress"])
    assert stored["status"] == "complete"
    assert stored["workers"] == 4
    assert stored["worker_limit"] == 4


def test_non_exact_error_stops_immediately_and_records_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, ["fails", "must_not_run"])
    calls: list[str] = []

    def fake_build_from_plan(**kwargs: Any) -> list[dict[str, Any]]:
        experiment_id = next(iter(kwargs["requested_experiments"]))
        calls.append(experiment_id)
        raise RuntimeError("Unverified or stale GT")

    monkeypatch.setattr(driver.standard_builder, "build_from_plan", fake_build_from_plan)

    result = _run(paths)

    assert result["status"] == "error"
    assert calls == ["fails"]
    assert result["experiments"]["fails"]["status"] == "error"
    assert result["experiments"]["must_not_run"]["status"] == "pending"
    error_event = result["experiments"]["fails"]["history"][-1]
    assert error_event["error_type"] == "RuntimeError"
    assert error_event["error"] == "Unverified or stale GT"


def test_resume_skips_base_and_audits_existing_rescue_without_rewriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, ["base_done", "rescue_done"])
    base_manifest = paths["base_root"] / "base_done" / "manifest.json"
    merged_manifest = paths["merged_root"] / "rescue_done" / "manifest.json"
    atomic_write_json(base_manifest, {"base": True})
    atomic_write_json(merged_manifest, {"merged": True})
    base_hash = sha256_file(base_manifest)
    merged_hash = sha256_file(merged_manifest)
    monkeypatch.setattr(
        driver.standard_builder,
        "build_from_plan",
        lambda **_kwargs: pytest.fail("resume called builder"),
    )
    monkeypatch.setattr(
        driver,
        "merge_experiment",
        lambda **_kwargs: pytest.fail("resume called merge"),
    )
    audit_calls: list[str] = []

    def fake_audit(*, experiment_id: str, contract_path: Path) -> dict[str, Any]:
        audit_calls.append(experiment_id)
        return {"status": "ok", "experiment_id": experiment_id, "errors": []}

    monkeypatch.setattr(driver, "audit_merged_experiment", fake_audit)

    result = _run(paths)

    assert result["status"] == "complete"
    assert result["experiments"]["base_done"]["status"] == (
        "base_export_already_complete"
    )
    assert result["experiments"]["rescue_done"]["status"] == (
        "rescue_already_complete"
    )
    assert audit_calls == ["rescue_done"]
    assert sha256_file(base_manifest) == base_hash
    assert sha256_file(merged_manifest) == merged_hash
