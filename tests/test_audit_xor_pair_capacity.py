from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from architecture_iq.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "audit_xor_pair_capacity", ROOT / "tools" / "audit_xor_pair_capacity.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_candidate(root: Path, candidate_id: str, model_type: str, mean: float) -> None:
    path = root / candidate_id
    path.mkdir(parents=True)
    (path / "candidate_spec.json").write_text(
        json.dumps({"candidate_id": candidate_id, "model": {"type": model_type}}),
        encoding="utf-8",
    )
    summary = {
        "candidate_id": candidate_id,
        "failed_seeds": 0,
        "excluded": False,
        "mean_test_ce": mean,
        "std_test_ce": 0.001,
        "seed_results": [
            {"seed": index, "failed": False, "final_test_ce": mean}
            for index in range(10)
        ],
    }
    (path / "results").mkdir()
    (path / "results" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_audit_deduplicates_and_counts_significant_cross_family_pairs(tmp_path: Path) -> None:
    first = tmp_path / "set_a"
    second = tmp_path / "set_b"
    first.mkdir()
    second.mkdir()
    _write_candidate(first, "c_mlp", "mlp", 0.20)
    _write_candidate(first, "c_kan", "kan", 0.30)
    # Duplicate id is ignored, even when encountered in another set.
    _write_candidate(second, "c_kan", "kan", 0.31)

    report = _MODULE.audit_xor_pair_capacity(
        [first, second], load_profile("v2.3-xor-pilot"), target_significant=1, question_count=1
    )

    assert report["candidate_count"] == 2
    assert report["total_pairs"] == 1
    assert report["significant_pairs"] == 1
    assert report["mlp_wins"] == 1
    assert report["kan_wins"] == 0
    assert report["target_significant_reached"] is True
    assert report["can_select_question_count"] is False
    assert report["duplicate_candidate_ids"][0]["candidate_id"] == "c_kan"


def test_audit_rejects_nonzero_failures_and_nonfinite_seed_metric(tmp_path: Path) -> None:
    root = tmp_path / "set"
    root.mkdir()
    _write_candidate(root, "c_mlp", "mlp", 0.20)
    _write_candidate(root, "c_kan", "kan", 0.30)
    summary_path = root / "c_kan" / "results" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["failed_seeds"] = 1
    summary["seed_results"][0]["final_test_ce"] = "nan"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = _MODULE.audit_xor_pair_capacity([root], load_profile("v2.3-xor-pilot"))

    assert report["total_pairs"] == 0
    assert report["invalid_candidate_count"] == 1
    assert report["invalid_candidates"][0]["candidate_id"] == "c_kan"


@pytest.mark.parametrize(
    ("mlp", "kan", "expected"),
    [(70, 30, True), (71, 29, False), (50, 0, False)],
)
def test_question_capacity_fraction(mlp: int, kan: int, expected: bool) -> None:
    ok, _ = _MODULE._can_select_question_count(100, mlp, kan, 0.70)
    assert ok is expected
