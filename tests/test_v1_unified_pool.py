from __future__ import annotations

from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, list_dataset_families


def test_v1_enables_all_families_and_compatible_models() -> None:
    ensure_registries()
    profile = load_profile("v1")
    assert set(profile.pools["dataset_families"]) == set(list_dataset_families())
    expected = {
        "univariate_regression": {"mlp", "kan"},
        "multivariate_regression": {"mlp", "kan"},
        "bigram_lm": {"transformer_lm", "gru_lm"},
        "synthetic_tabular_classification": {"mlp", "kan"},
    }
    for family_name, models in expected.items():
        family = get_dataset_family(family_name)
        assert set(family.compatible_model_types()) == models
        assert (
            set(profile.model_types_for_family(family_name, family.compatible_model_types()))
            == models
        )
