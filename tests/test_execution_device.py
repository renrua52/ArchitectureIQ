from __future__ import annotations

from architecture_iq.candidates.axes import choices_compatible
from architecture_iq.candidates.generator import (
    CLASSIFICATION_TRAIN_PY,
    build_candidate_spec,
)
from architecture_iq.ground_truth.runner import _resolve_execution_device
from architecture_iq.profile import load_profile


def _spec(*, device: str, width: int = 16) -> dict:
    profile = load_profile("v1")
    return build_candidate_spec(
        profile,
        dataset_id="sym_device_test",
        family="univariate_regression",
        budget=1024,
        batch_size=16,
        model={
            "type": "mlp",
            "depth": 2,
            "width": width,
            "residual": False,
            "layer_norm": [False, False],
            "activations": ["relu", "relu"],
        },
        optimizer={
            "type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        loss={"loss_id": "mse"},
        execution_device=device,
    )


def test_frozen_v1_profile_keeps_legacy_hash_and_cpu_default() -> None:
    profile = load_profile("v1")
    assert "device" not in profile.ground_truth
    assert profile.execution_device == "cpu"
    assert profile.profile_hash == "164f68c29f6730dc"


def test_legacy_candidate_without_execution_is_always_cpu() -> None:
    profile = load_profile("v1")
    profile.ground_truth["device"] = "cuda"
    assert str(_resolve_execution_device({}, profile)) == "cpu"


def test_device_changes_candidate_identity_and_mixed_choices_are_rejected() -> None:
    cpu = _spec(device="cpu", width=16)
    cuda = _spec(device="cuda", width=32)
    assert cpu["candidate_id"] != cuda["candidate_id"]
    assert not choices_compatible([cpu, cuda])


def test_classification_generated_train_loop_is_device_aware() -> None:
    assert 'device: str = "cpu"' in CLASSIFICATION_TRAIN_PY
    assert "Model().to(run_device)" in CLASSIFICATION_TRAIN_PY
    assert "device=run_device" in CLASSIFICATION_TRAIN_PY