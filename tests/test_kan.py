from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.families.univariate_regression import UnivariateRegressionFamily
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import load_profile
from architecture_iq.prompts.code_excerpt import excerpt_model_py
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train


def _kan_spec() -> dict:
    return {
        "type": "kan",
        "variant": "efficient_spline_v1",
        "input_dim": 1,
        "output_dim": 1,
        "depth": 1,
        "width": 4,
        "grid_size": 3,
        "spline_order": 3,
        "grid_range": [-1.0, 1.0],
        "base_activation": "silu",
    }


def test_v1_is_kan_free_and_v2_exposes_kan() -> None:
    v1 = load_profile("v1")
    v2 = load_profile("v2")
    assert "kan" not in v1.pools["model_types"]
    assert "kan" in v2.pools["model_types"]
    assert v2.kan["variant"] == "efficient_spline_v1"


def test_kan_forward_and_backward() -> None:
    ensure_registries()
    model = get_model_type("kan").build_module(_kan_spec())
    x = torch.rand(8, 1, requires_grad=True)
    output = model(x)
    assert output.shape == (8, 1)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_rendered_kan_executes_through_ground_truth() -> None:
    ensure_registries()
    profile = load_profile("v2")
    profile.ground_truth["n_seeds"] = 2
    profile.ground_truth["max_failed_seeds"] = 2
    family = get_dataset_family("univariate_regression")
    assert isinstance(family, UnivariateRegressionFamily)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        partial = family.create_instance(profile, 123)
        dataset_spec = family.build_spec_with_id(partial)
        dataset_path = root / "dataset"
        family.materialize({**partial, **dataset_spec}, dataset_path)
        candidate_spec = build_candidate_spec(
            profile,
            dataset_id=dataset_spec["dataset_id"],
            family="univariate_regression",
            budget=64,
            batch_size=16,
            model=_kan_spec(),
            optimizer={
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            loss={"loss_id": "mse"},
        )
        candidate_path = root / "candidate"
        write_candidate(candidate_spec, candidate_path, get_model_type("kan"))
        train_module = load_candidate_train(candidate_path)
        train_x, _, _, _ = family.load_tensors(dataset_path)
        assert train_module.Model()(train_x[:4]).shape == (4, 1)

        summary = run_ground_truth(
            candidate_path,
            profile,
            dataset_path,
            fail_threshold_override=float("inf"),
        )
        assert summary["execution"] == "candidate_py_files"
        assert summary["selection_metric"] == "test_mse"
        assert summary["failed_seeds"] == 0
        assert torch.isfinite(torch.tensor(summary["mean_test_mse"]))


def test_kan_model_excerpt_includes_executed_helpers() -> None:
    ensure_registries()
    source = get_model_type("kan").render_model_py(_kan_spec())
    excerpt = excerpt_model_py(source)
    assert "def _make_grid" in excerpt
    assert "def _bspline_bases" in excerpt
    assert "class KANLinear" in excerpt
