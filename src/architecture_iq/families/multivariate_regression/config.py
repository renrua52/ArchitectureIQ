"""Resolve multivariate input dimension from profile pool or explicit override."""

from __future__ import annotations

import random
from typing import Any

from architecture_iq.profile import Profile


def allowed_input_dims(profile: Profile) -> list[int]:
    cfg = profile.family_config("multivariate_regression")
    return [int(v) for v in cfg.get("input_dims", [4])]


def resolve_input_dim(
    profile: Profile,
    *,
    input_dim: int | None = None,
    rng: random.Random | None = None,
) -> int:
    allowed = allowed_input_dims(profile)
    if input_dim is not None:
        if input_dim not in allowed:
            raise ValueError(
                f"input_dim must be one of {allowed} for multivariate_regression "
                f"(got {input_dim})"
            )
        return int(input_dim)
    picker = rng if rng is not None else random.Random()
    return int(picker.choice(allowed))
