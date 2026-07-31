from __future__ import annotations

from tools.meta_model_dataset.supplemental_reserve_common import (
    BASE_PLAN_PATH,
    CONTRACT_ID,
    SUPPLEMENTAL_PLAN_PATH,
    deterministic_seed,
    phase_experiments,
    read_json,
    sha256_file,
    validate_static_plans,
)
from tools.meta_model_dataset.merge_supplemental_reserve import (
    RescueNotRequired,
    _activation_failure,
    _enrich_replacements,
    _tier2_reserves,
)
from tools.meta_model_dataset.core import select_usable_rows

import pytest


BASE_PLAN_SHA256 = "8ba994e1ba2ac168fa6f193911bfdc02de089358cfeca1764983d74b68af9338"


def test_supplemental_plan_is_a_deterministic_context_preserving_b2_copy() -> None:
    base_plan, supplemental_plan, policy = validate_static_plans()

    assert sha256_file(BASE_PLAN_PATH) == BASE_PLAN_SHA256
    assert supplemental_plan["base_plan_sha256"] == BASE_PLAN_SHA256
    assert policy["contract_id"] == CONTRACT_ID
    assert len(phase_experiments(base_plan)) == 21
    assert len(phase_experiments(supplemental_plan)) == 21
    assert supplemental_plan["defaults"]["num_rows"] == 50
    assert supplemental_plan["defaults"]["train_rows"] == 45
    assert supplemental_plan["defaults"]["reserve_rows"] == 17


def test_supplemental_seeds_and_base_manifest_exclusions_are_explicit() -> None:
    plan = read_json(SUPPLEMENTAL_PLAN_PATH)

    for experiment in plan["experiments"]:
        experiment_id = experiment["experiment_id"]
        assert experiment["sampling_seed"] == deterministic_seed(
            BASE_PLAN_SHA256, experiment_id, "sampling"
        )
        assert experiment["split_seed"] == deterministic_seed(
            BASE_PLAN_SHA256, experiment_id, "split"
        )
        assert experiment["exclude_sampling_manifests"] == [
            "data/meta_model/setting_to_loss_wide_v2/"
            f"{experiment_id}/sampling_manifest.json"
        ]


def test_policy_forbids_target_based_or_cross_split_rescue() -> None:
    _base, _supplemental, policy = validate_static_plans()

    assert policy["activation"]["message_prefix"] == (
        "Not enough usable reserve settings to replace "
    )
    assert policy["selection"]["permitted_label_gate"] == (
        "usable_for_regression only"
    )
    assert policy["selection"]["cross_split_allowed"] is False
    assert policy["selection"]["replacement_order"] == [
        "same_split_and_stratum",
        "same_split",
    ]
    forbidden = policy["selection"]["forbidden_selection_fields"]
    assert "target.mean_loss" in forbidden
    assert "target.benchmark_eligible" in forbidden


def _attempt(
    fingerprint: str,
    *,
    role: str,
    split: str = "train",
    stratum: str = "s",
    usable: bool = True,
    sampling_index: int = 0,
) -> dict:
    return {
        "example_fingerprint_sha256": fingerprint,
        "selection_role": role,
        "split": split,
        "stratum": stratum,
        "usable_for_regression": usable,
        "sampling_index": sampling_index,
    }


def test_activation_is_only_the_exact_base_capacity_failure() -> None:
    policy = read_json(
        SUPPLEMENTAL_PLAN_PATH.with_name("supplemental_reserve_policy_v1.json")
    )
    prefix = policy["activation"]["message_prefix"]
    attempts = [
        _attempt("bad", role="primary", usable=False),
    ]

    message = _activation_failure(
        attempts,
        train_rows=1,
        validation_rows=0,
        message_prefix=prefix,
    )

    assert message.startswith(prefix)
    with pytest.raises(RescueNotRequired):
        _activation_failure(
            [_attempt("good", role="primary")],
            train_rows=1,
            validation_rows=0,
            message_prefix=prefix,
        )


def test_tier2_reserve_prefers_same_stratum_and_never_crosses_split() -> None:
    base_attempts = [
        _attempt("bad", role="primary", usable=False, stratum="wanted"),
        _attempt("base-fallback", role="reserve", stratum="other"),
    ]
    side_selected = [
        {
            **_attempt(
                "supp-exact",
                role="primary",
                stratum="wanted",
                sampling_index=4,
            ),
            "dataset_role": "primary",
        }
    ]
    tier2 = _tier2_reserves(side_selected, base_attempt_count=len(base_attempts))

    assert tier2[0]["selection_role"] == "reserve"
    assert tier2[0]["split"] == "train"
    assert tier2[0]["supplemental_origin"]["selection_role"] == "primary"
    selected, replacements = select_usable_rows(
        [*base_attempts, *tier2],
        train_rows=1,
        validation_rows=0,
    )
    enriched = _enrich_replacements(replacements, [*base_attempts, *tier2])

    assert selected[0]["example_fingerprint_sha256"] == "supp-exact"
    assert enriched[0]["source_tier"] == "supplemental_v1"
    assert enriched[0]["replacement_route"] == "same_split_and_stratum"
