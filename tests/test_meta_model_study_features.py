from __future__ import annotations

import json
import math
from copy import deepcopy

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from tools.meta_model_study.features import (
    FEATURE_SETS,
    FeatureEncoder,
    feature_dict,
    load_jsonl,
)


def _mlp_example(
    *,
    params: int = 201,
    optimizer: str = "Adam",
    lr: float = 0.001,
    width: int = 8,
    activations: list[str] | None = None,
    layer_norm: list[bool] | None = None,
) -> dict:
    setting = {
        "model": {
            "type": "mlp",
            "input_dim": 3,
            "depth": 2,
            "width": width,
            "residual": True,
            "activations": activations or ["relu", "gelu"],
            "layer_norm": layer_norm or [True, False],
        },
        "optimizer": {
            "type": optimizer,
            "lr": lr,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
        },
        "loss": {"loss_id": "mse"},
        "budget": {
            "batch_size": 32,
            "training_steps": 64,
            "total_samples_seen": 2048,
        },
    }
    return {
        "setting": setting,
        "derived": {
            "total_params": params,
            "trainable_params": params,
            "log_total_params": math.log(params),
        },
        # These fields must never enter the feature dictionary.
        "target": {"mean_loss": 0.25},
        "provenance": {"candidate_path": "private"},
    }


def test_load_jsonl_accepts_blank_lines_and_reports_bad_lines(tmp_path) -> None:
    valid = tmp_path / "rows.jsonl"
    valid.write_text('{"value": 1}\n\n {"value": 2}\n', encoding="utf-8")
    assert load_jsonl(valid) == [{"value": 1}, {"value": 2}]

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"value": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        load_jsonl(invalid)

    non_object = tmp_path / "list.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_jsonl(non_object)


def test_feature_sets_only_use_log_parameter_count() -> None:
    row = _mlp_example()

    params = feature_dict(row, "params")
    assert params == {"derived.log_total_params": pytest.approx(math.log(201))}
    for feature_set in FEATURE_SETS:
        features = feature_dict(row, feature_set)
        assert "derived.total_params" not in features
        assert "derived.trainable_params" not in features
        assert "target.mean_loss" not in features
        assert features["derived.log_total_params"] == pytest.approx(math.log(201))

    total_only = deepcopy(row)
    del total_only["derived"]["log_total_params"]
    assert feature_dict(total_only, "params") == params

    inconsistent = deepcopy(row)
    inconsistent["derived"]["log_total_params"] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        feature_dict(inconsistent, "params")


def test_parameter_count_can_be_strictly_excluded() -> None:
    row = _mlp_example()
    row["derived"] = {}

    compact = feature_dict(row, "compact", include_parameter_count=False)

    assert "derived.log_total_params" not in compact
    assert compact["optimizer.type"] == "Adam"
    with pytest.raises(ValueError, match="requires parameter count"):
        feature_dict(row, "params", include_parameter_count=False)

    encoder = FeatureEncoder(
        feature_set="compact",
        include_parameter_count=False,
    ).fit([row, _mlp_example(width=16)])
    assert all("param" not in name for name in encoder.feature_names)

    changed = deepcopy(row)
    changed["derived"] = {
        "total_params": 10**9,
        "trainable_params": 10**9,
        "log_total_params": math.log(10**9),
    }
    assert np.array_equal(encoder.transform([row]), encoder.transform([changed]))


def test_optimizer_interaction_and_mlp_summaries_are_explicit() -> None:
    row = _mlp_example(
        activations=["relu", "relu", "gelu"],
        layer_norm=[True, False, True],
    )
    optimizer_features = feature_dict(row, "optimizer_lr")
    compact = feature_dict(row, "compact")
    full = feature_dict(row, "full")

    assert optimizer_features["optimizer.type"] == "Adam"
    assert optimizer_features["optimizer.log10_lr"] == pytest.approx(-3.0)
    assert optimizer_features["interaction.optimizer_x_log10_lr.Adam"] == pytest.approx(
        -3.0
    )
    assert compact["model.activation_count.relu"] == 2
    assert compact["model.activation_fraction.relu"] == pytest.approx(2.0 / 3.0)
    assert compact["model.activation_unique_count"] == 2
    assert compact["model.layer_norm_count"] == 2
    assert compact["model.layer_norm_fraction"] == pytest.approx(2.0 / 3.0)
    assert compact["model.layer_norm_transitions"] == 2
    assert full["model.activations[2]"] == "gelu"
    assert full["model.layer_norm[1]"] == 0
    assert full["optimizer.lr"] == pytest.approx(0.001)


def test_compact_transformer_features_capture_structure() -> None:
    example = {
        "setting": {
            "model": {
                "type": "transformer_lm",
                "vocab_size": 32,
                "context_length": 16,
                "d_model": 64,
                "num_layers": 3,
                "num_heads": 4,
                "d_ff": 256,
            },
            "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
            "loss": {"loss_id": "cross_entropy"},
            "budget": {
                "batch_size": 64,
                "training_steps": 80,
                "total_samples_seen": 5120,
            },
        },
        "derived": {"total_params": 100_000},
    }

    compact = feature_dict(example, "compact")

    assert compact["model.d_model"] == 64
    assert compact["model.d_ff"] == 256
    assert compact["model.num_layers"] == 3
    assert compact["model.num_heads"] == 4
    assert compact["model.d_ff_over_d_model"] == pytest.approx(4.0)
    assert compact["model.head_dim"] == pytest.approx(16.0)
    assert compact["model.layers_x_d_model"] == pytest.approx(192.0)


def test_encoder_fits_only_train_vocabulary_mask_and_scaler_and_round_trips(
    tmp_path,
) -> None:
    train = [
        _mlp_example(params=101, optimizer="Adam", lr=0.001, width=8),
        _mlp_example(params=201, optimizer="AdamW", lr=0.003, width=16),
        _mlp_example(params=401, optimizer="SGD", lr=0.01, width=32),
        _mlp_example(
            params=801,
            optimizer="RMSprop",
            lr=0.0003,
            width=64,
            activations=["silu", "relu"],
        ),
    ]
    external = _mlp_example(
        params=1601,
        optimizer="NeverSeen",
        lr=0.0001,
        width=128,
        activations=["never_seen", "never_seen"],
    )
    encoder = FeatureEncoder(feature_set="full")

    matrix = encoder.fit_transform(np.asarray(train, dtype=object))
    vocabulary_before = dict(encoder.vectorizer_.vocabulary_)
    mean_before = encoder.scaler_.mean_.copy()
    names_before = encoder.feature_names.copy()
    external_matrix = encoder.transform([external])

    assert matrix.shape == (4, len(encoder.feature_names))
    assert external_matrix.shape == (1, matrix.shape[1])
    assert np.all(np.isfinite(external_matrix))
    assert encoder.vectorizer_.vocabulary_ == vocabulary_before
    assert np.array_equal(encoder.scaler_.mean_, mean_before)
    assert encoder.feature_names == names_before
    assert all("NeverSeen" not in name for name in encoder.feature_names)
    assert all("never_seen" not in name for name in encoder.feature_names)
    # The fixed training budget is removed using training variance alone.
    assert "budget.batch_size" not in encoder.feature_names

    artifact = tmp_path / "encoder.joblib"
    joblib.dump(encoder, artifact)
    restored = joblib.load(artifact)
    assert restored.feature_names == encoder.feature_names
    assert np.array_equal(restored.transform([external]), external_matrix)


def test_encoder_is_cloneable_and_operates_inside_sklearn_pipeline() -> None:
    rows = [
        _mlp_example(params=100 + index * 50, lr=10 ** (-4 + index / 3))
        for index in range(8)
    ]
    targets = np.linspace(-2.0, 1.0, len(rows))
    encoder = FeatureEncoder(feature_set="optimizer_lr")

    cloned = clone(encoder)
    assert cloned.feature_set == "optimizer_lr"
    assert cloned.include_parameter_count is True
    assert not hasattr(cloned, "vectorizer_")

    pipeline = Pipeline([("features", encoder), ("model", Ridge(alpha=1.0))])
    pipeline.fit(np.asarray(rows, dtype=object), targets)
    prediction = pipeline.predict(np.asarray(rows, dtype=object))
    assert prediction.shape == targets.shape
    assert np.all(np.isfinite(prediction))


def test_unknown_feature_set_and_invalid_input_fail_clearly() -> None:
    row = _mlp_example()
    with pytest.raises(ValueError, match="Unknown feature_set"):
        feature_dict(row, "mystery")
    with pytest.raises(TypeError, match="mapping"):
        FeatureEncoder("params").fit(["not a row"])

    path = json.dumps(row)  # Also ensure rows are JSON serializable fixtures.
    assert isinstance(path, str)
def test_dataset_conditioning_modes_and_unseen_values_are_safe() -> None:
    first = _mlp_example()
    first["dataset_context"] = {
        "dataset_id": "dataset_a",
        "description": {"family": "regression", "params": {"degree": 2}},
    }
    second = deepcopy(first)
    second["dataset_context"] = {
        "dataset_id": "dataset_b",
        "description": {"family": "regression", "params": {"degree": 3}},
    }

    assert not any(name.startswith("dataset.") for name in feature_dict(first, "compact"))
    assert feature_dict(first, "compact", dataset_conditioning="id")["dataset.id"] == "dataset_a"
    described = feature_dict(first, "compact", dataset_conditioning="description")
    assert described["dataset.description.family"] == "regression"
    assert described["dataset.description.params.degree"] == 2

    id_encoder = FeatureEncoder("compact", dataset_conditioning="id").fit([first, second])
    unseen_id = deepcopy(first)
    unseen_id["dataset_context"]["dataset_id"] = "never_seen"
    assert id_encoder.transform([unseen_id]).shape[0] == 1

    description_encoder = FeatureEncoder(
        "compact", dataset_conditioning="description"
    ).fit([first, second])
    unseen_description = deepcopy(first)
    unseen_description["dataset_context"]["description"] = {
        "family": "language_model",
        "params": {"vocab_size": 99},
    }
    assert description_encoder.transform([unseen_description]).shape[0] == 1
