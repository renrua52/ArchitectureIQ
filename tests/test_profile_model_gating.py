from __future__ import annotations

import random

from architecture_iq.candidates.generator import sample_model
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family


def test_v2_profile_hash_and_classification_gate_are_unchanged() -> None:
    ensure_registries()
    profile = load_profile("v2")
    family = get_dataset_family("synthetic_tabular_classification")

    assert profile.profile_hash == "3993a8aef680d37c"
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["mlp"]
    assert {
        sample_model(profile, random.Random(seed), family=family.name)["type"]
        for seed in range(32)
    } == {"mlp"}


def test_v21_explicitly_opens_classification_kan_gate() -> None:
    ensure_registries()
    profile = load_profile("v2.1")
    family = get_dataset_family("synthetic_tabular_classification")

    assert profile.profile_hash != "3993a8aef680d37c"
    assert profile.name == "v2.1"
    assert profile.kan["depth"] == [1, 2]
    assert profile.kan["width"] == [8]
    assert profile.kan["grid_size"] == [5]
    assert set(
        profile.model_types_for_family(family.name, family.compatible_model_types())
    ) == {"mlp", "kan"}
    sampled = {
        sample_model(profile, random.Random(seed), family=family.name)["type"]
        for seed in range(128)
    }
    assert sampled == {"mlp", "kan"}
