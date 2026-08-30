"""Registry <-> profile wiring for dataset families."""

from __future__ import annotations

import pytest

from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, list_dataset_families

# The profile that is expected to keep step with the registry. Registering a
# family and forgetting to wire it into this profile would leave it unreachable
# from the CLI, so that mismatch has to fail here. Older profiles are records of
# what produced earlier builds and are allowed to lag behind (v1 and v1.3 both
# predate xor_classification / spiral_classification), so they get the weaker
# check in test_profile_pools_only_list_registered_families.
ACTIVE_PROFILE = "v1.4"

LAGGING_PROFILES = ("v1", "v1.3")

EXPECTED_MODEL_TYPES = {
    "univariate_regression": {"mlp"},
    "multivariate_regression": {"mlp"},
    "bigram_lm": {"transformer_lm", "gru_lm"},
    "synthetic_tabular_classification": {"mlp"},
    "xor_classification": {"mlp"},
    "spiral_classification": {"mlp"},
}


def test_active_profile_enables_every_registered_family() -> None:
    ensure_registries()
    profile = load_profile(ACTIVE_PROFILE)
    assert set(profile.pools["dataset_families"]) == set(list_dataset_families())


def test_active_profile_configures_every_family_it_enables() -> None:
    """A family in the pool needs a dataset config and a loss pool to be sampleable."""
    ensure_registries()
    profile = load_profile(ACTIVE_PROFILE)
    for family_name in profile.pools["dataset_families"]:
        assert profile.family_config(family_name), family_name
        assert profile.pools["losses"][family_name], family_name


@pytest.mark.parametrize("profile_name", (ACTIVE_PROFILE, *LAGGING_PROFILES))
def test_profile_pools_only_list_registered_families(profile_name: str) -> None:
    ensure_registries()
    profile = load_profile(profile_name)
    assert set(profile.pools["dataset_families"]) <= set(list_dataset_families())


def test_registered_families_have_the_expected_model_compatibility() -> None:
    ensure_registries()
    assert set(EXPECTED_MODEL_TYPES) == set(list_dataset_families())
    for family_name, models in EXPECTED_MODEL_TYPES.items():
        family = get_dataset_family(family_name)
        assert set(family.compatible_model_types()) == models


@pytest.mark.parametrize("profile_name", (ACTIVE_PROFILE, *LAGGING_PROFILES))
def test_profile_model_pools_resolve_to_the_family_compatibility(profile_name: str) -> None:
    ensure_registries()
    profile = load_profile(profile_name)
    for family_name in profile.pools["dataset_families"]:
        family = get_dataset_family(family_name)
        resolved = profile.model_types_for_family(family_name, family.compatible_model_types())
        assert set(resolved) == EXPECTED_MODEL_TYPES[family_name]
