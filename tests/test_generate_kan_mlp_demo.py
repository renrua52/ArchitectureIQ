from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from architecture_iq.profile import load_profile
from generate_kan_mlp_demo import _fixed_training, _load_pair_config, _pair_config_hash, _pair_specs


def test_default_external_pair_config_injects_dataset_dimension_and_fixed_training() -> None:
    profile = load_profile("v2")
    config = _load_pair_config(profile)
    pairs = _pair_specs(config, input_dim=7)

    assert [name for name, _, _ in pairs] == [
        "kan_d1_w4_g3__mlp_d1_w11",
        "kan_d1_w8_g5__mlp_d1_w18",
        "kan_d2_w8_g5__mlp_d2_w24",
    ]
    for _, kan, mlp in pairs:
        assert kan["input_dim"] == 7
        assert mlp["input_dim"] == 7
        assert kan["output_dim"] == mlp["output_dim"] == 1

    batch_size, optimizer, loss = _fixed_training(config)
    assert batch_size == 32
    assert optimizer == {"type": "Adam", "lr": 0.003, "weight_decay": 0.0, "betas": [0.9, 0.999]}
    assert loss == {"loss_id": "mse"}


def test_explicit_pair_config_accepts_wrapped_yaml() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "kan_mlp_pairs_v2.1.yaml"
    profile = load_profile("v2")
    config = _load_pair_config(profile, config_path)
    pairs = _pair_specs(config, input_dim=3)

    assert pairs[0][0] == "kan_d1_w4_g3__mlp_d1_w11"
    assert pairs[0][1]["input_dim"] == pairs[0][2]["input_dim"] == 3
    assert _fixed_training(config)[0] == 32
    assert len(_pair_config_hash(config_path)) == 64
