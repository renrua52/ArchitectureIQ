from __future__ import annotations

import random

from architecture_iq.candidates.generator import sample_model
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family


def test_v2_classification_models_are_mlp_only() -> None:
    ensure_registries()
    profile = load_profile("v2")
    family = get_dataset_family("synthetic_tabular_classification")

    # KAN was removed from code and pools; classification is MLP-only.
    assert "kan" not in profile.raw
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["mlp"]
    assert {
        sample_model(profile, random.Random(seed), family=family.name)["type"]
        for seed in range(128)
    } == {"mlp"}


def test_v21_v22_classification_models_are_mlp_only() -> None:
    ensure_registries()
    family = get_dataset_family("synthetic_tabular_classification")

    for name in ("v2.1", "v2.2"):
        profile = load_profile(name)
        assert profile.name == name
        assert "kan" not in profile.raw
        assert profile.model_types_for_family(
            family.name, family.compatible_model_types()
        ) == ["mlp"]
        sampled = {
            sample_model(profile, random.Random(seed), family=family.name)["type"]
            for seed in range(128)
        }
        assert sampled == {"mlp"}


def test_v23_gru_pilot_opens_bigram_gru_gate() -> None:
    ensure_registries()
    profile = load_profile("v2.3-gru-pilot")
    family = get_dataset_family("bigram_lm")

    assert profile.name == "v2.3-gru-pilot"
    assert "kan" not in profile.raw
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["transformer_lm", "gru_lm"]


def test_v24_xor_review_profile_contract() -> None:
    ensure_registries()
    profile = load_profile("v2.4-xor-review")
    family = get_dataset_family("synthetic_tabular_classification")

    assert profile.name == "v2.4-xor-review"
    assert "kan" not in profile.raw
    assert profile.family_config("synthetic_tabular_classification")["rule_families"] == ["xor"]
    assert profile.significance == {
        "gap_min": 0.0,
        "win_rate_min": 0.7,
        "use_non_overlap": True,
    }
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["mlp"]
