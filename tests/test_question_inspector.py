"""Tests for the standalone question inspector loader."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import from tools directory without installing a package.
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))

from artifact_loader import (  # noqa: E402
    format_metrics,
    list_question_dirs,
    load_question_bundle,
    load_dataset_tensors,
    question_label,
)
from candidate_curves import all_step_samples, load_candidate_curves, reconstruct_eval_samples  # noqa: E402
from custom_settings import (  # noqa: E402
    build_custom_setting_spec,
    build_loss_spec,
    build_model_spec,
    build_optimizer_spec,
    custom_setting_run_id,
    form_values_from_candidate_spec,
    list_custom_setting_runs,
    question_custom_settings_dir,
    run_custom_setting,
)
from expression_latex import expression_to_latex, expression_to_mathml  # noqa: E402
from architecture_iq.profile import load_profile  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "quiz_demo" / "bundle"


@pytest.fixture
def question_path() -> Path:
    questions = list_question_dirs(DATA)
    assert questions, "the bundled demo must contain at least one question"
    return questions[0]


def test_expression_to_latex() -> None:
    latex = expression_to_latex("cos(6.283185307179586*(0.8333 + x)) + (x - x + x / -2)")
    assert r"\cos" in latex
    assert "x" in latex


def test_expression_to_latex_powers() -> None:
    latex = expression_to_latex("sin(6.283185307179586*x) + (x)**2")
    assert r"\sin" in latex
    assert "2" in latex


def test_expression_to_latex_multivariate_subscripts() -> None:
    latex = expression_to_latex("x0 + sin(6.283185307179586*x1) * x2")
    assert "x_{0}" in latex or r"x_0" in latex
    assert "x_{1}" in latex or r"x_1" in latex
    assert "x_{2}" in latex or r"x_2" in latex
    assert "3 x" not in latex


def test_expression_to_mathml_multivariate_subscripts() -> None:
    mathml = expression_to_mathml("x0 + sin(6.283185307179586*x1) * x2")
    assert mathml.startswith('<math xmlns="http://www.w3.org/1998/Math/MathML"')
    assert "<msub>" in mathml
    assert "π" in mathml


def test_load_question_bundle(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    assert bundle.question["question_id"].startswith("q_")
    assert len(bundle.choices) == bundle.question["num_choices"]
    for choice in bundle.choices:
        assert choice["candidate_dir"].is_dir()
        assert (choice["candidate_dir"] / "candidate_spec.json").is_file()


def test_load_dataset_tensors(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    tx, ty, vx, vy = load_dataset_tensors(bundle.dataset_dir)
    assert tx.ndim == 2 and ty.shape == tx.shape
    assert vx.shape == tx.shape and vy.shape == ty.shape


def test_format_metrics() -> None:
    text = format_metrics({"selection_metric": "test_mse", "mean_test_mse": 0.1, "std_test_mse": 0.02})
    assert "0.100000" in text
    assert "0.020000" in text


def test_all_step_samples() -> None:
    assert all_step_samples(2048, 32) == [32 * step for step in range(1, 65)]
    assert all_step_samples(1024, 64) == [64 * step for step in range(1, 17)]


def test_reconstruct_eval_samples_sparse() -> None:
    assert reconstruct_eval_samples(2048, 32, 64) == [
        32,
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        576,
        640,
        704,
        768,
        832,
        896,
        960,
        1024,
        1088,
        1152,
        1216,
        1280,
        1344,
        1408,
        1472,
        1536,
        1600,
        1664,
        1728,
        1792,
        1856,
        1920,
        1984,
        2048,
    ]
    assert reconstruct_eval_samples(1024, 64, 64) == [
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        576,
        640,
        704,
        768,
        832,
        896,
        960,
        1024,
    ]


def _regression_dataset_spec() -> dict:
    return {
        "schema_version": "1.0",
        "family": "multivariate_regression",
        "dataset_id": "mvar_test",
        "params": {"input_dim": 4},
        "selection_metric": "test_mse",
    }


def test_build_custom_setting_spec() -> None:
    profile = load_profile("v1")
    dataset_spec = _regression_dataset_spec()
    model = build_model_spec(
        "mlp",
        {
            "depth": 2,
            "width": 48,
            "residual": True,
            "activations": ["relu", "gelu"],
            "layer_norm": [False, True],
        },
        dataset_spec["params"],
    )
    optimizer = build_optimizer_spec(
        "AdamW",
        lr=2e-3,
        weight_decay=1e-4,
        betas=(0.8, 0.99),
    )
    loss = build_loss_spec("mse_l2", lambda_value=5e-4)
    spec = build_custom_setting_spec(
        profile,
        dataset_spec,
        budget=960,
        batch_size=32,
        model=model,
        optimizer=optimizer,
        loss=loss,
    )

    assert spec["budget"]["training_steps"] == 30
    assert spec["model"]["input_dim"] == 4
    assert spec["model"]["activations"] == ["relu", "gelu"]
    assert spec["optimizer"]["betas"] == [0.8, 0.99]
    assert spec["loss"]["lambda"] == 5e-4


def test_inherited_form_values_rebuild_exact_candidate_spec() -> None:
    profile = load_profile("v1")
    dataset_spec = _regression_dataset_spec()
    source_model = build_model_spec(
        "mlp",
        {
            "depth": 3,
            "width": 128,
            "residual": True,
            "activations": ["relu", "gelu", "silu"],
            "layer_norm": [True, False, True],
        },
        dataset_spec["params"],
    )
    source_optimizer = build_optimizer_spec(
        "AdamW",
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.85, 0.995),
    )
    source_loss = build_loss_spec("mse_l1", lambda_value=1e-2)
    source = build_custom_setting_spec(
        profile,
        dataset_spec,
        budget=2048,
        batch_size=32,
        model=source_model,
        optimizer=source_optimizer,
        loss=source_loss,
    )

    values = form_values_from_candidate_spec(
        source,
        source_letter="B",
        evaluation={"n_seeds": 10, "base_seed": 7},
    )
    rebuilt_model = build_model_spec(
        values["model_type"],
        {
            "depth": values["mlp_depth"],
            "width": values["mlp_width"],
            "residual": values["mlp_residual"],
            "activations": [
                values[f"mlp_activation_{index}"]
                for index in range(values["mlp_depth"])
            ],
            "layer_norm": [
                values[f"mlp_norm_{index}"]
                for index in range(values["mlp_depth"])
            ],
        },
        dataset_spec["params"],
    )
    rebuilt_optimizer = build_optimizer_spec(
        values["optimizer_type"],
        lr=values["learning_rate"],
        weight_decay=values["weight_decay"],
        betas=(values["beta1"], values["beta2"]),
    )
    rebuilt = build_custom_setting_spec(
        profile,
        dataset_spec,
        budget=values["budget"],
        batch_size=values["batch_size"],
        model=rebuilt_model,
        optimizer=rebuilt_optimizer,
        loss=build_loss_spec(values["loss"], lambda_value=values["loss_lambda"]),
    )

    assert values["n_seeds"] == 10
    assert values["base_seed"] == 7
    assert rebuilt == source
    assert rebuilt["candidate_id"] == source["candidate_id"]


def test_build_custom_setting_rejects_invalid_budget() -> None:
    profile = load_profile("v1")
    dataset_spec = _regression_dataset_spec()
    model = build_model_spec(
        "mlp",
        {
            "depth": 1,
            "width": 16,
            "residual": False,
            "activations": ["relu"],
            "layer_norm": [False],
        },
        dataset_spec["params"],
    )
    with pytest.raises(ValueError, match="divisible"):
        build_custom_setting_spec(
            profile,
            dataset_spec,
            budget=100,
            batch_size=32,
            model=model,
            optimizer=build_optimizer_spec("SGD", lr=1e-3, weight_decay=0),
            loss=build_loss_spec("mse"),
        )


def test_transformer_setting_validates_attention_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_model_spec(
            "transformer_lm",
            {"d_model": 30, "num_layers": 1, "num_heads": 8, "d_ff": 64},
            {"vocab_size": 32, "context_length": 16},
        )


def test_run_custom_setting_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_settings
    import numpy as np

    profile = load_profile("v1")
    dataset_spec = _regression_dataset_spec()
    model = build_model_spec(
        "mlp",
        {
            "depth": 1,
            "width": 16,
            "residual": False,
            "activations": ["relu"],
            "layer_norm": [False],
        },
        dataset_spec["params"],
    )
    spec = build_custom_setting_spec(
        profile,
        dataset_spec,
        budget=64,
        batch_size=16,
        model=model,
        optimizer=build_optimizer_spec("Adam", lr=1e-3, weight_decay=0),
        loss=build_loss_spec("mse"),
    )

    def fake_ground_truth(
        candidate_path,
        run_profile,
        dataset_path,
        *,
        fail_threshold_override,
    ):
        assert run_profile.n_seeds == 2
        assert run_profile.base_seed == 11
        assert fail_threshold_override == float("inf")
        results = candidate_path / "results"
        results.mkdir(parents=True)
        np.savez(
            results / "curves.npz",
            curves=np.asarray([[1.0, 0.5], [0.9, 0.4]]),
            samples=np.asarray([16, 32]),
        )
        return {"selection_metric": "test_mse", "mean_test_mse": 0.45}

    monkeypatch.setattr(custom_settings, "run_ground_truth", fake_ground_truth)
    question_root = tmp_path / "q_test"
    storage_root = question_custom_settings_dir(question_root)
    dataset_path = tmp_path / "dataset"
    result = run_custom_setting(
        storage_root,
        dataset_path,
        profile,
        spec,
        label="My setting",
        n_seeds=2,
        base_seed=11,
    )

    expected_id = custom_setting_run_id(
        spec,
        n_seeds=2,
        base_seed=11,
        sequence=1,
    )
    assert result["custom_setting_id"] == expected_id
    assert result["candidate_dir"].parent == storage_root
    runs = list_custom_setting_runs(storage_root)
    assert len(runs) == 1
    assert runs[0]["label"] == "My setting #0001"


def test_custom_settings_keep_latest_and_best_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import custom_settings
    import numpy as np

    profile = load_profile("v1")
    dataset_spec = _regression_dataset_spec()
    model = build_model_spec(
        "mlp",
        {
            "depth": 1,
            "width": 16,
            "residual": False,
            "activations": ["relu"],
            "layer_norm": [False],
        },
        dataset_spec["params"],
    )
    spec = build_custom_setting_spec(
        profile,
        dataset_spec,
        budget=64,
        batch_size=16,
        model=model,
        optimizer=build_optimizer_spec("Adam", lr=1e-3, weight_decay=0),
        loss=build_loss_spec("mse"),
    )
    metrics = iter([0.5, 0.3, 0.4, 0.2])

    def fake_ground_truth(
        candidate_path,
        run_profile,
        dataset_path,
        *,
        fail_threshold_override,
    ):
        metric = next(metrics)
        results = candidate_path / "results"
        results.mkdir(parents=True)
        np.savez(
            results / "curves.npz",
            curves=np.asarray([[1.0, metric]]),
            samples=np.asarray([16, 32]),
        )
        return {"selection_metric": "test_mse", "mean_test_mse": metric}

    monkeypatch.setattr(custom_settings, "run_ground_truth", fake_ground_truth)
    question_root = tmp_path / "q_test"
    storage_root = question_custom_settings_dir(question_root)
    dataset_path = tmp_path / "dataset"
    results = [
        run_custom_setting(
            storage_root,
            dataset_path,
            profile,
            spec,
            label="Trial",
            n_seeds=1,
            base_seed=0,
        )
        for _ in range(4)
    ]

    assert len({result["custom_setting_id"] for result in results}) == 4
    assert [result["label"] for result in results] == [
        "Trial #0001",
        "Trial #0002",
        "Trial #0003",
        "Trial #0004",
    ]
    retained = list_custom_setting_runs(storage_root)
    assert [run["label"] for run in retained] == ["Trial #0004", "Trial #0002"]
    assert [run["final_metric"] for run in retained] == [0.2, 0.3]
    setting_dirs = [
        path
        for path in storage_root.iterdir()
        if path.is_dir()
    ]
    assert len(setting_dirs) == 2


def test_load_candidate_curves(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    choice = bundle.choices[0]
    curves_path = choice["candidate_dir"] / "results" / "curves.npz"
    if not curves_path.is_file():
        pytest.skip("no curves.npz for candidate")
    spec_path = choice["candidate_dir"] / "candidate_spec.json"
    import json

    spec = json.loads(spec_path.read_text())
    budget = spec["budget"]
    loaded = load_candidate_curves(
        curves_path,
        total_samples_seen=int(budget["total_samples_seen"]),
        batch_size=int(budget["batch_size"]),
    )
    assert "error" not in loaded
    assert loaded["curves"].ndim == 2
    assert len(loaded["eval_samples"]) == loaded["curves"].shape[1]


def test_list_question_dirs() -> None:
    pool = list_question_dirs(DATA)
    assert pool
    assert all(p.name.startswith("q_") for p in pool)
    assert all((p / "question.json").is_file() for p in pool)


def test_question_label(question_path: Path) -> None:
    label = question_label(question_path)
    assert question_path.name in label
    assert " · " in label
