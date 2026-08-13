from __future__ import annotations

import math
import random

import pytest
import torch

from architecture_iq.candidates.generator import (
    build_candidate_spec,
    sample_model,
    trainable_parameter_count,
    write_candidate,
)
from architecture_iq.families.bigram_lm import BigramLmFamily
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from tools.analyze_order_parameters import count_params


def _model_spec() -> dict[str, int | str]:
    return {
        "type": "gru_lm",
        "vocab_size": 7,
        "context_length": 4,
        "d_model": 5,
        "num_layers": 2,
    }


def test_legacy_specs_default_to_non_residual() -> None:
    ensure_registries()
    family = get_model_type("gru_lm")
    legacy = _model_spec()
    model = family.build_module(legacy)
    assert model.layer_residual is False
    assert hasattr(model, "gru")
    assert not hasattr(model, "gru_layers")


def test_residual_forward_has_identity_path_and_same_parameter_count() -> None:
    ensure_registries()
    family = get_model_type("gru_lm")
    spec = {**_model_spec(), "layer_residual": True}
    model = family.build_module(spec)
    assert model.layer_residual is True
    assert len(model.gru_layers) == spec["num_layers"]
    assert not hasattr(model, "gru")
    for parameter in model.gru_layers.parameters():
        parameter.data.zero_()
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    expected = model.head(model.token_embed(tokens))
    torch.testing.assert_close(model(tokens), expected)
    assert trainable_parameter_count(spec) == trainable_parameter_count(_model_spec())

def test_registry_forward_and_backward() -> None:
    ensure_registries()
    family = get_model_type("gru_lm")
    model = family.build_module(_model_spec())
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 4, 7)
    logits.square().mean().backward()
    assert next(model.parameters()).grad is not None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("vocab_size", 1, "vocab_size"),
        ("context_length", 0, "context_length"),
        ("d_model", 0, "d_model"),
        ("num_layers", 0, "num_layers"),
    ],
)
def test_invalid_specs_are_rejected(key: str, value: int, message: str) -> None:
    spec = _model_spec()
    spec[key] = value
    with pytest.raises(ValueError, match=message):
        get_model_type("gru_lm").validate(spec)


def test_causality() -> None:
    model = get_model_type("gru_lm").build_module(_model_spec()).eval()
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    changed = tokens.clone()
    changed[0, 3] = 6
    original_logits = model(tokens)
    changed_logits = model(changed)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])


def test_default_profile_includes_gru_for_bigram() -> None:
    ensure_registries()
    family = get_dataset_family("bigram_lm")
    assert family.compatible_model_types() == ["transformer_lm", "gru_lm"]
    # Legacy v2-series profiles keep transformer-only bigram pools unless gated.
    for name in ("v2", "v2.1", "v2.2"):
        assert "gru_lm" not in load_profile(name).pools["model_types"]
    v1 = load_profile("v1")
    assert set(v1.model_types_for_family("bigram_lm", family.compatible_model_types())) == {
        "transformer_lm",
        "gru_lm",
    }
    pilot = load_profile("v2.3-gru-pilot")
    assert set(pilot.model_types_for_family("bigram_lm", family.compatible_model_types())) == {
        "transformer_lm",
        "gru_lm",
    }
    sampled = sample_model(
        v1,
        random.Random(2),
        family="bigram_lm",
        dataset_params={"vocab_size": 7, "context_length": 4},
    )
    assert sampled["type"] in {"transformer_lm", "gru_lm"}


def test_residual_profile_samples_all_gru_specs_with_residuals() -> None:
    profile = load_profile("v2.5-gru-residual-architecture-pilot")
    bridge = profile.raw["cross_profile_reuse"]
    assert bridge["enabled"] is True
    assert bridge["transformer_allowlist"] == [
        {
            "source_profile": "v2.4-gru-architecture-pilot",
            "source_profile_hash": "f0893f8ab4ec7cf0",
            "model_types": ["transformer_lm"],
            "note": bridge["transformer_allowlist"][0]["note"],
        }
    ]
    assert profile.significance["gap_min"] == 0
    assert profile.significance["win_rate_min"] == 0.7
    assert profile.significance["use_non_overlap"] is True
    assert len(profile.gru_lm["d_model"]) * len(profile.gru_lm["num_layers"]) == 104
    assert profile.gru_lm["layer_residual"] is True
    from architecture_iq.models.gru_lm import GruLmModelFamily
    sampled = [
        GruLmModelFamily().sample_spec(
            profile, random.Random(seed), {"vocab_size": 7, "context_length": 4}
        )
        for seed in range(64)
    ]
    assert all(spec["layer_residual"] is True for spec in sampled)
    assert {spec["d_model"] for spec in sampled} <= set(profile.gru_lm["d_model"])
    assert {spec["num_layers"] for spec in sampled} <= set(profile.gru_lm["num_layers"])


def test_rendered_residual_model_is_executable() -> None:
    ensure_registries()
    family = get_model_type("gru_lm")
    rendered = family.render_model_py({**_model_spec(), "layer_residual": True})
    assert "self.gru_layers = nn.ModuleList" in rendered
    assert "h = h + layer_output" in rendered
    namespace: dict[str, object] = {}
    exec(rendered, namespace)
    model = namespace["Model"]()
    logits = model(torch.zeros((1, 4), dtype=torch.long))
    assert logits.shape == (1, 4, 7)

def test_architecture_pilot_profile_uses_adam_and_large_gru_pool() -> None:
    profile = load_profile("v2.4-gru-architecture-pilot")
    assert profile.pools["dataset_families"] == ["bigram_lm"]
    assert profile.pools["model_types"] == ["transformer_lm", "gru_lm"]
    assert profile.pools["optimizers"] == ["Adam"]
    assert profile.pools["losses"]["bigram_lm"] == ["cross_entropy"]
    assert profile.optimizer_grids["lr"] == [1.0e-3]
    assert profile.optimizer_grids["weight_decay"] == [0.0]
    assert profile.optimizer_grids["batch_size"] == [32]
    assert len(profile.gru_lm["d_model"]) * len(profile.gru_lm["num_layers"]) >= 100


def test_parameter_audit_matches_module() -> None:
    model_spec = _model_spec()
    expected = 2 * 7 * 5 + 2 * (6 * 5 * 5 + 6 * 5) + 7
    assert count_params(model_spec) == expected
    actual = sum(
        parameter.numel()
        for parameter in get_model_type("gru_lm").build_module(model_spec).parameters()
    )
    assert trainable_parameter_count(model_spec) == actual == expected


def test_core_prompt_renderer_describes_gru() -> None:
    from architecture_iq.prompts.formatters import format_model_spec_lines

    text = "\n".join(format_model_spec_lines(_model_spec()))
    assert "causal unidirectional GRU" in text
    assert "No attention" in text


def test_rendered_candidate_reuses_lm_trainer_and_smoke_runs(tmp_path) -> None:
    ensure_registries()
    profile = load_profile("v2.3-gru-pilot")
    profile.ground_truth["n_seeds"] = 1
    family = BigramLmFamily()
    partial = family.create_instance(profile, seed=11)
    partial["params"].update(
        {"vocab_size": 7, "context_length": 4, "train_size": 16, "test_size": 8}
    )
    dataset_spec = family.build_spec_with_id(partial)
    dataset_path = tmp_path / "dataset"
    family.materialize({**partial, **dataset_spec}, dataset_path)
    model = _model_spec()
    candidate_spec = build_candidate_spec(
        profile,
        dataset_id=dataset_spec["dataset_id"],
        family="bigram_lm",
        budget=8,
        batch_size=4,
        model=model,
        optimizer={"type": "Adam", "lr": 1e-3, "weight_decay": 0.0},
        loss={"loss_id": "cross_entropy"},
    )
    candidate_path = tmp_path / "candidate"
    write_candidate(candidate_spec, candidate_path, get_model_type("gru_lm"))
    train_module = load_candidate_train(candidate_path)
    assert hasattr(train_module, "train_and_eval")
    assert train_module.Model()(torch.zeros((1, 4), dtype=torch.long)).shape == (1, 4, 7)
    summary = run_ground_truth(candidate_path, profile, dataset_path, sync_files=False)
    assert summary["execution"] == "candidate_py_files"
    assert summary["n_seeds"] == 1
    assert math.isfinite(summary["mean_test_ce"])
