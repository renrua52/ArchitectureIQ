from __future__ import annotations

import importlib.util
from pathlib import Path

from architecture_iq.registry import ensure_registries
from architecture_iq.profile import load_profile
import architecture_iq.questions.generator as question_generator
from architecture_iq.questions.generator import _pick_balanced_unique_pairs
from architecture_iq.significance.validator import SignificanceResult


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "xor_sampled_pool", ROOT / "tools" / "build_xor_sampled_pool.py"
)
assert SPEC and SPEC.loader
POOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POOL)


_DATASET_SPEC = {
    "family": "synthetic_tabular_classification",
    "params": {"input_dim": 2, "num_classes": 2, "rule_family": "xor"},
}


def test_family_specific_sampler_freezes_16_unique_specs_per_family(tmp_path: Path) -> None:
    ensure_registries()
    matrix = POOL.sample_matrix(
        load_profile("v2.5-xor-holdout"),
        dataset_spec=_DATASET_SPEC,
        mlp_seed=11,
        kan_seed=12,
        count=16,
    )

    pool = matrix["candidate_pool"]
    mlps = [entry["model"] for entry in pool if entry["model"]["type"] == "mlp"]
    kans = [entry["model"] for entry in pool if entry["model"]["type"] == "kan"]
    assert len(mlps) == len(kans) == 16
    assert len({POOL._canonical_model(model) for model in mlps}) == 16
    assert len({POOL._canonical_model(model) for model in kans}) == 16
    assert all(model["input_dim"] == 2 and model["output_dim"] == 2 for model in [*mlps, *kans])
    assert matrix["shared_training"] == {
        "total_samples_seen": 8192,
        "batch_size": 32,
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "loss": {"loss_id": "cross_entropy"},
    }

    path = tmp_path / "sampled.yaml"
    digest = POOL.write_matrix(matrix, path)
    loaded, loaded_digest = POOL.load_matrix(path)
    assert loaded == matrix
    assert loaded_digest == digest


def test_supplemental_one_family_sampling_excludes_frozen_specs() -> None:
    ensure_registries()
    profile = load_profile("v2.5-xor-holdout")
    initial = POOL.sample_matrix(
        profile,
        dataset_spec=_DATASET_SPEC,
        mlp_seed=11,
        kan_seed=12,
        count=4,
    )
    frozen_mlps = {
        POOL._canonical_model(entry["model"])
        for entry in initial["candidate_pool"]
        if entry["model"]["type"] == "mlp"
    }
    supplemental = POOL.sample_matrix(
        profile,
        dataset_spec=_DATASET_SPEC,
        mlp_seed=13,
        kan_seed=12,
        mlp_count=4,
        kan_count=0,
        excluded_model_keys={"mlp": frozen_mlps},
        excluded_matrix_hashes=["frozen-matrix-sha256"],
    )

    supplemental_models = [entry["model"] for entry in supplemental["candidate_pool"]]
    assert len(supplemental_models) == 4
    assert {model["type"] for model in supplemental_models} == {"mlp"}
    assert not frozen_mlps.intersection(
        POOL._canonical_model(model) for model in supplemental_models
    )
    assert supplemental["sampling"]["models_by_family"] == {"mlp": 4, "kan": 0}
    assert supplemental["sampling"]["excluded_matrix_hashes"] == ["frozen-matrix-sha256"]

def test_winner_cap_selects_unique_pairs_with_no_family_above_seventy_percent(
    monkeypatch,
) -> None:
    subsets: list[list[Path]] = []
    winner_types: dict[frozenset[str], str] = {}
    model_types: dict[Path, str] = {}
    for index in range(10):
        pair = [Path(f"mlp_{index}"), Path(f"kan_{index}")]
        subsets.append(pair)
        model_types[pair[0]] = "mlp"
        model_types[pair[1]] = "kan"
        winner_types[frozenset(path.name for path in pair)] = (
            "kan" if index < 7 else "mlp"
        )

    monkeypatch.setattr(question_generator, "load_summary", lambda path: path)

    def fake_significance(summaries, _profile, *, metric):
        winner_type = winner_types[frozenset(path.name for path in summaries)]
        winner_index = next(
            index
            for index, path in enumerate(summaries)
            if model_types[path] == winner_type
        )
        return SignificanceResult(True, 0.0, 1.0, metric, winner_index)

    monkeypatch.setattr(
        question_generator,
        "validate_significance",
        fake_significance,
    )
    selected = _pick_balanced_unique_pairs(
        subsets,
        10,
        profile=load_profile("v2.5-xor-holdout"),
        selection_metric="test_cross_entropy",
        model_types=model_types,
        max_winner_fraction=0.7,
    )

    assert len(selected) == 10
    assert len({frozenset(path.name for path in pair) for pair in selected}) == 10
    selected_winners = [
        winner_types[frozenset(path.name for path in pair)] for pair in selected
    ]
    assert selected_winners.count("kan") == 7
    assert selected_winners.count("mlp") == 3
