from __future__ import annotations

import importlib.util
from pathlib import Path

from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "xor_matrix_materializer", ROOT / "tools" / "materialize_xor_candidate_matrix.py"
)
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_xor_matrix_resolves_exact_two_dimensional_models() -> None:
    ensure_registries()
    matrix, digest = MATRIX.load_matrix(ROOT / "configs" / "xor_architecture_matrix_v1.yaml")
    comparisons = MATRIX.resolve_comparisons(
        matrix,
        dataset_spec={
            "family": "synthetic_tabular_classification",
            "params": {"input_dim": 2, "num_classes": 2, "rule_family": "xor"},
        },
        profile=load_profile("v2.4-xor-review"),
    )

    assert len(digest) == 64
    assert [comparison_id for comparison_id, _ in comparisons] == ["xor_relu_mlp_vs_spline_kan"]
    models = comparisons[0][1]
    assert [model["type"] for model in models] == ["mlp", "kan"]
    assert all(model["input_dim"] == 2 and model["output_dim"] == 2 for model in models)


def test_xor_matrix_rejects_non_xor_dataset() -> None:
    matrix, _ = MATRIX.load_matrix(ROOT / "configs" / "xor_architecture_matrix_v1.yaml")
    try:
        MATRIX.resolve_comparisons(
            matrix,
            dataset_spec={
                "family": "synthetic_tabular_classification",
                "params": {"input_dim": 2, "num_classes": 2, "rule_family": "smooth_additive"},
            },
            profile=load_profile("v2.4-xor-review"),
        )
    except ValueError as exc:
        assert "rule_family" in str(exc)
    else:
        raise AssertionError("expected matrix dataset validation to fail")


def test_xor_v2_batch01_is_a_full_exact_matrix() -> None:
    ensure_registries()
    matrix, digest = MATRIX.load_matrix(ROOT / "configs" / "xor_architecture_matrix_v2_batch01.yaml")
    comparisons = MATRIX.resolve_comparisons(
        matrix,
        dataset_spec={
            "family": "synthetic_tabular_classification",
            "params": {"input_dim": 2, "num_classes": 2, "rule_family": "xor"},
        },
        profile=load_profile("v2.5-xor-screen"),
    )

    assert len(digest) == 64
    assert len(comparisons) == 24
    assert comparisons[0][0].startswith("b01_c01_")
    assert all([model["type"] for model in models] == ["mlp", "kan"] for _, models in comparisons)
    for _, models in comparisons:
        mlp = models[0]
        assert len(mlp["activations"]) == mlp["depth"]
        assert len(mlp["layer_norm"]) == mlp["depth"]