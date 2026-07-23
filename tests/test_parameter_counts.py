from __future__ import annotations

from copy import deepcopy

from architecture_iq.candidates.generator import build_candidate_spec, trainable_parameter_count
from architecture_iq.profile import load_profile
from architecture_iq.util import short_hash


def _spec(*, width: int = 16) -> dict:
    return build_candidate_spec(
        load_profile("v1"),
        dataset_id="sym_parameter_count",
        family="univariate_regression",
        budget=1024,
        batch_size=32,
        model={
            "type": "mlp",
            "input_dim": 1,
            "output_dim": 1,
            "depth": 2,
            "width": width,
            "residual": False,
            "layer_norm": [False, False],
            "activations": ["relu", "relu"],
        },
        optimizer={"type": "Adam", "lr": 1e-3, "weight_decay": 0.0},
        loss={"loss_id": "mse"},
    )


def test_new_candidate_spec_hashes_trainable_parameter_count() -> None:
    spec = _spec()
    assert spec["trainable_parameter_count"] == trainable_parameter_count(spec["model"])

    hashed_body = deepcopy(spec)
    candidate_id = hashed_body.pop("candidate_id")
    assert candidate_id == f"c_{short_hash(hashed_body)}"

    hashed_body["trainable_parameter_count"] += 1
    assert candidate_id != f"c_{short_hash(hashed_body)}"


def test_model_size_changes_new_candidate_identity() -> None:
    assert _spec(width=16)["candidate_id"] != _spec(width=32)["candidate_id"]