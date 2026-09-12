"""adam_betas as a sampleable pool, without disturbing older profiles.

Before v1.4 the field was one flat ``[beta1, beta2]`` pair that ``sample_optimizer``
read without touching the RNG. Making it a pool has to stay invisible to the 13
profiles carrying the flat form: if sampling drew from the RNG unconditionally,
every Adam/AdamW candidate downstream of that draw would shift and the recorded
candidate ids of existing artifacts would no longer reproduce.
"""

from __future__ import annotations

import random

import pytest
import yaml

from architecture_iq.candidates.generator import sample_optimizer
from architecture_iq.paths import PROFILES_DIR
from architecture_iq.profile import Profile, load_profile


def _profile_with_betas(betas: object) -> Profile:
    raw = yaml.safe_load((PROFILES_DIR / "v1.4.yaml").read_text(encoding="utf-8"))
    raw["optimizer_grids"]["adam_betas"] = betas
    return Profile(
        raw=raw,
        name=raw["profile"],
        schema_version=raw["schema_version"],
        pools=raw["pools"],
        dataset=raw["dataset"],
        mlp=raw["mlp"],
        optimizer_grids=raw["optimizer_grids"],
        loss_grids=raw.get("loss_grids", {}),
        budgets=raw["budgets"],
        training_defaults=raw.get("training_defaults", {}),
        ground_truth=raw["ground_truth"],
        significance=raw["significance"],
        question_generation=raw["question_generation"],
        prompts=raw["prompts"],
    )


def test_flat_pair_reads_as_a_single_option() -> None:
    assert load_profile("v1.3").adam_betas_pool() == [(0.9, 0.999)]


def test_v14_carries_both_pairs() -> None:
    assert load_profile("v1.4").adam_betas_pool() == [(0.9, 0.999), (0.9, 0.95)]


@pytest.mark.parametrize(
    "betas",
    [[], [0.9], [0.9, 0.99, 0.999], [[0.9, 0.999], [0.9]], [[0.9, 0.999], 0.95]],
)
def test_malformed_shapes_are_rejected(betas: object) -> None:
    with pytest.raises(ValueError):
        _profile_with_betas(betas).adam_betas_pool()


def test_a_single_pair_pool_consumes_no_randomness() -> None:
    """The regression guard: one fixed pair must leave the RNG where it was."""
    profile = _profile_with_betas([0.9, 0.999])
    for seed in range(64):
        left, right = random.Random(seed), random.Random(seed)
        spec = sample_optimizer(profile, left)
        # Replay every draw sample_optimizer makes *except* a betas draw.
        opt_type = right.choice(profile.pools["optimizers"])
        right.choice(profile.optimizer_grids["lr"])
        right.choice(profile.optimizer_grids["weight_decay"])
        if opt_type == "SGD":
            right.choice(profile.optimizer_grids["sgd_momentum"])
        assert spec["type"] == opt_type
        assert left.random() == right.random()


def test_a_multi_pair_pool_actually_varies_betas() -> None:
    profile = load_profile("v1.4")
    seen = {
        tuple(sample_optimizer(profile, random.Random(seed)).get("betas", ()))
        for seed in range(300)
    }
    assert {(0.9, 0.999), (0.9, 0.95)} <= seen


def test_only_adam_family_optimizers_get_betas() -> None:
    profile = load_profile("v1.4")
    for seed in range(300):
        spec = sample_optimizer(profile, random.Random(seed))
        assert ("betas" in spec) == (spec["type"] in {"Adam", "AdamW"})
