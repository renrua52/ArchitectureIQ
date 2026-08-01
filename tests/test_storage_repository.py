"""Tests for the backend storage layer and the legacy->columnar migration."""
from __future__ import annotations

import json
from pathlib import Path

from architecture_iq.storage import repository as repo
from architecture_iq.storage import schema as sc
from tools.storage.migrate_data_layout import migrate

REGRESSION_TRAIN_PY = '"""Training loop for this candidate."""\nfrom loss import loss_fn\n'


def _make_legacy_tree(root: Path) -> Path:
    """Build a small legacy data/datasets tree (1 family, 2 datasets)."""
    root.mkdir(parents=True)

    def make_dataset(family: str, dataset_id: str, metric: str, extra: list[str]) -> Path:
        d = root / family / dataset_id
        (d / "candidates" / f"set_100_var_fix_fix_abc123").mkdir(parents=True)
        spec = {
            "schema_version": "1.0",
            "family": family,
            "dataset_id": dataset_id,
            "params": {"instance_seed": 1, "noise": {"enabled": False}},
            "selection_metric": metric,
            "significance": {"gap_min": 0.05, "fail_threshold": 2.0},
            "files": {"synthesize": "synthesize.py", "train": "train.pt", "test": "test.pt", **{f: f for f in extra}},
        }
        (d / "dataset_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (d / "synthesize.py").write_text("def synthesize():\n    pass\n", encoding="utf-8")
        (d / "train.pt").write_bytes(b"train-bytes")
        (d / "test.pt").write_bytes(b"test-bytes")
        for f in extra:
            (d / f).write_bytes(b"extra-bytes")
        for cid in ("c_a1", "c_b2"):
            c = d / "candidates" / f"set_100_var_fix_fix_abc123" / cid
            c.mkdir(parents=True)
            cand = {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "family": family,
                "budget": {"training_steps": 100, "batch_size": 1, "total_samples_seen": 100},
                "model": {"type": "mlp", "depth": 2, "width": 8},
                "optimizer": {"type": "SGD", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.0},
                "loss": {"loss_id": "mse"},
                "files": {"model": "model.py", "train": "train.py", "loss": "loss.py", "optimizer": "optimizer.py"},
                "candidate_id": cid,
            }
            (c / "candidate_spec.json").write_text(json.dumps(cand), encoding="utf-8")
            (c / "train.py").write_text(REGRESSION_TRAIN_PY, encoding="utf-8")
            (c / "results").mkdir()
            (c / "results" / "summary.json").write_text(
                json.dumps({"candidate_id": cid, "selection_metric": metric,
                            f"mean_{metric}": 0.5, "n_seeds": 10}),
                encoding="utf-8",
            )
            (c / "results" / "curves.npz").write_bytes(b"curves")
        (d / "questions" / "run_1q_2c_zzz").mkdir(parents=True)
        return d

    make_dataset("multivariate_regression", "mvar_a1b2", "test_mse", extra=[])
    make_dataset("bigram_lm", "bg_9f8e", "test_ce", extra=["transition.npz"])
    return root


def test_migrate_copy(tmp_path: Path) -> None:
    src = _make_legacy_tree(tmp_path / "src")
    dest = tmp_path / "dest"
    counts = migrate(src, dest, "copy", families=None, limit=None)

    assert counts["problems"] == 2
    assert counts["candidates"] == 4
    assert counts["results"] == 4
    assert counts["trainers"] == 2  # one per family
    assert counts["skipped_questions"] == 2

    # problems carry spec + files + README
    pspec = json.loads((dest / "problems" / "mvar_a1b2" / "dataset_spec.json").read_text())
    assert pspec["schema_version"] == sc.PROBLEM_SCHEMA_VERSION
    assert pspec["problem_id"] == "mvar_a1b2"
    assert (dest / "problems" / "mvar_a1b2" / "synthesize.py").exists()
    assert (dest / "problems" / "mvar_a1b2" / "train.pt").read_bytes() == b"train-bytes"
    assert (dest / "problems" / "bg_9f8e" / "transition.npz").read_bytes() == b"extra-bytes"
    assert (dest / "problems" / "bg_9f8e" / "README.md").is_file()

    # candidate configs are closed JSON v2
    cfg = json.loads((dest / "candidates" / "mvar_a1b2" / "c_a1.json").read_text())
    assert cfg["schema_version"] == sc.CANDIDATE_SCHEMA_VERSION
    assert cfg["problem_id"] == "mvar_a1b2"
    assert cfg["candidate_id"] == "c_a1"
    assert "files" not in cfg

    # results mirror candidates
    assert (dest / "results" / "mvar_a1b2" / "c_a1" / "summary.json").is_file()
    assert (dest / "results" / "mvar_a1b2" / "c_a1" / "curves.npz").read_bytes() == b"curves"

    # trainers per family
    trainer = json.loads((dest / "trainers" / "multivariate_regression_v1" / "trainer_spec.json").read_text())
    assert trainer["trainer_id"] == "multivariate_regression_v1"
    assert (dest / "trainers" / "multivariate_regression_v1" / "train.py").read_text() == REGRESSION_TRAIN_PY

    # sources untouched in copy mode
    assert (src / "multivariate_regression" / "mvar_a1b2" / "candidates").is_dir()


def test_migrate_idempotent_and_dry_run(tmp_path: Path) -> None:
    src = _make_legacy_tree(tmp_path / "src")
    dest = tmp_path / "dest"

    dry = migrate(src, dest, "dry-run", families=None, limit=None)
    assert dry["problems"] == 2
    assert not (dest / "problems").exists()

    first = migrate(src, dest, "copy", families=None, limit=None)
    second = migrate(src, dest, "copy", families=None, limit=None)
    assert first == second
    assert sorted(p.name for p in (dest / "candidates" / "mvar_a1b2").glob("*.json")) == ["c_a1.json", "c_b2.json"]


def test_repository_api(tmp_path: Path) -> None:
    src = _make_legacy_tree(tmp_path / "src")
    dest = tmp_path / "dest"
    migrate(src, dest, "copy", families=None, limit=None)

    assert repo.list_problems() == ["bg_9f8e", "mvar_a1b2"]
    assert repo.list_candidate_ids("mvar_a1b2") == ["c_a1", "c_b2"]
    assert repo.list_results("mvar_a1b2") == ["c_a1", "c_b2"]
    assert repo.list_trainers() == ["bigram_lm_v1", "multivariate_regression_v1"]
    summary = repo.read_summary("mvar_a1b2", "c_a1")
    assert summary["candidate_id"] == "c_a1"
    spec = repo.read_problem_spec("bg_9f8e")
    assert spec["selection_metric"] == "test_ce"
    assert repo.curves_path("mvar_a1b2", "c_a1").name == sc.CURVES_NPZ


def test_schema_validation() -> None:
    assert sc.validate_candidate_config({
        "schema_version": "2.0", "problem_id": "p", "candidate_id": "c",
        "family": "f", "budget": {}, "model": {}, "optimizer": {}, "loss": {},
    }) == []
    assert "candidate_id" in sc.validate_candidate_config({"schema_version": "2.0"})
    assert sc.validate_problem_spec({
        "schema_version": "2.0", "problem_id": "p", "dataset_id": "p", "family": "f",
        "params": {}, "selection_metric": "m", "files": {},
    }) == []
