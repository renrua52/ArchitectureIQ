from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import json

from architecture_iq.candidates.generator import choices_compatible, sample_variant_pool
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries
from architecture_iq.significance.validator import SignificanceResult
from architecture_iq.questions.generator import (
    build_question_record,
    _pick_candidate_disjoint_subsets,
    eligible_candidate_paths,
    find_significant_subsets,
    select_significant_candidates,
)


def test_sample_variant_pool_unique() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(0)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        question_type="architecture_only",
        pool_size=8,
        rng=rng,
    )
    assert len(specs) == 8
    assert len({s["candidate_id"] for s in specs}) == 8


def test_architecture_only_shares_batch_size() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(0)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        question_type="architecture_only",
        pool_size=8,
        rng=rng,
    )
    batch_sizes = {spec["budget"]["batch_size"] for spec in specs}
    assert len(batch_sizes) == 1
    optimizers = {spec["optimizer"]["type"] for spec in specs}
    assert len(optimizers) == 1
    assert choices_compatible(specs, "architecture_only")


def test_mixed_pool_varies_all_training_axes() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(42)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=8192,
        question_type="mixed",
        pool_size=24,
        rng=rng,
    )
    assert len(specs) == 24
    models = {json.dumps(spec["model"], sort_keys=True) for spec in specs}
    optimizers = {json.dumps(spec["optimizer"], sort_keys=True) for spec in specs}
    losses = {json.dumps(spec["loss"], sort_keys=True) for spec in specs}
    batch_sizes = {spec["budget"]["batch_size"] for spec in specs}
    assert len(models) > 1
    assert len(optimizers) > 1
    assert len(losses) > 1
    assert len(batch_sizes) == 1


def test_choices_compatible_rejects_mixed_batch_sizes() -> None:
    specs = [
        {
            "budget": {"batch_size": 16},
            "model": {"depth": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
        {
            "budget": {"batch_size": 32},
            "model": {"depth": 2},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
    ]
    assert not choices_compatible(specs, "architecture_only")


def _fake_candidate_specs() -> dict[str, dict]:
    base = {
        "budget": {"batch_size": 16, "total_samples_seen": 1024, "training_steps": 64},
        "optimizer": {"type": "Adam"},
        "loss": {"loss_id": "mse"},
    }
    return {
        "good": {**base, "model": {"depth": 1}, "candidate_id": "good"},
        "mid": {**base, "model": {"depth": 2}, "candidate_id": "mid"},
        "bad1": {**base, "model": {"depth": 3}, "candidate_id": "bad1"},
        "bad2": {**base, "model": {"depth": 4}, "candidate_id": "bad2"},
    }


def test_profile_default_num_choices() -> None:
    profile = load_profile("v1")
    assert profile.num_choices == 2


def test_find_significant_subsets_returns_multiple() -> None:
    profile = load_profile("v1")

    def fake_load_summary(path: Path) -> dict:
        means = {
            "good": (0.1, 0.01, [0.09] * 10),
            "mid": (0.5, 0.02, [0.5] * 10),
            "bad1": (0.7, 0.02, [0.7] * 10),
            "bad2": (0.8, 0.02, [0.8] * 10),
        }
        key = path.name
        mean, std, finals = means[key]
        return {
            "excluded": False,
            "mean_test_mse": mean,
            "std_test_mse": std,
            "seed_results": [{"failed": False, "final_test_mse": v} for v in finals],
        }

    pool = [Path(p) for p in ("good", "mid", "bad1", "bad2")]
    rng = __import__("random").Random(0)
    specs = _fake_candidate_specs()

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            subsets = find_significant_subsets(pool, profile, rng, num_choices=2)

    assert len(subsets) >= 2
    picked = _pick_candidate_disjoint_subsets(subsets, 2)
    assert len(picked) == 2
    assert set(picked[0]).isdisjoint(picked[1])


def test_candidate_disjoint_picker_backtracks_past_greedy_dead_end() -> None:
    subsets = [
        [Path("a"), Path("c")],
        [Path("a"), Path("b")],
        [Path("c"), Path("d")],
    ]

    picked = _pick_candidate_disjoint_subsets(subsets, 2)

    assert picked == [subsets[1], subsets[2]]


def test_candidate_disjoint_picker_returns_empty_when_impossible() -> None:
    subsets = [
        [Path("a"), Path("b")],
        [Path("a"), Path("c")],
        [Path("b"), Path("c")],
    ]

    assert _pick_candidate_disjoint_subsets(subsets, 2) == []


def test_select_significant_candidates_exhaustive() -> None:
    profile = load_profile("v1")

    def fake_load_summary(path: Path) -> dict:
        means = {
            "good": (0.1, 0.01, [0.09] * 10),
            "mid": (0.5, 0.02, [0.5] * 10),
            "bad1": (0.7, 0.02, [0.7] * 10),
            "bad2": (0.8, 0.02, [0.8] * 10),
        }
        key = path.name
        mean, std, finals = means[key]
        return {
            "excluded": False,
            "mean_test_mse": mean,
            "std_test_mse": std,
            "seed_results": [{"failed": False, "final_test_mse": v} for v in finals],
        }

    pool = [Path(p) for p in ("good", "mid", "bad1", "bad2")]
    rng = __import__("random").Random(0)
    specs = _fake_candidate_specs()

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            selected = select_significant_candidates(pool, profile, rng)

    assert selected is not None
    assert len(selected) == 2


def test_eligible_candidate_paths_filters_excluded() -> None:
    paths = [Path("a"), Path("b")]

    def fake_load_summary(path: Path) -> dict:
        return {"excluded": path.name == "b"}

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        assert eligible_candidate_paths(paths) == [Path("a")]


def test_build_question_record_persists_profile_hash(tmp_path: Path, monkeypatch) -> None:
    import architecture_iq.questions.generator as question_generator

    profile = load_profile("v1")
    data_root = tmp_path / "data"
    dataset_path = data_root / "datasets" / "univariate_regression" / "sym_test"
    set_path = dataset_path / "candidates" / "set_test"
    candidate_paths = [set_path / "c_one", set_path / "c_two"]
    model = {
        "type": "mlp",
        "depth": 1,
        "width": 16,
        "residual": False,
        "layer_norm": [False],
        "activations": ["relu"],
        "input_dim": 1,
        "output_dim": 1,
    }
    for index, candidate_path in enumerate(candidate_paths):
        candidate_path.mkdir(parents=True)
        (candidate_path / "candidate_spec.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_path.name,
                    "budget": {"training_steps": 64, "batch_size": 16, "total_samples_seen": 1024},
                    "model": {**model, "width": 16 + index},
                    "optimizer": {"type": "Adam"},
                    "loss": {"loss_id": "mse"},
                    "execution": {"device": "cpu"},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(question_generator, "DATA_DIR", data_root)
    monkeypatch.setattr(question_generator, "choices_compatible", lambda specs: True)
    monkeypatch.setattr(question_generator, "infer_question_type", lambda specs: "architecture_only")
    monkeypatch.setattr(question_generator, "infer_axes", lambda specs: (frozenset({"optimizer", "loss", "batch_size"}), frozenset({"model"})))
    monkeypatch.setattr(question_generator, "load_summary", lambda path: {"mean_test_mse": 0.1, "std_test_mse": 0.01})
    monkeypatch.setattr(
        question_generator,
        "validate_significance",
        lambda summaries, profile, metric: SignificanceResult(True, 0.2, 1.0, metric, 0),
    )

    record = build_question_record(
        profile,
        dataset_spec={
            "family": "univariate_regression",
            "dataset_id": "sym_test",
            "selection_metric": "test_mse",
        },
        dataset_path=dataset_path,
        candidate_paths=candidate_paths,
        candidate_set_paths=[set_path],
        rng=__import__("random").Random(0),
    )

    assert record["profile"] == "v1"
    assert record["profile_hash"] == profile.profile_hash
    assert record["question_id"].startswith("q_")

def test_candidate_disjoint_picker_large_pool_is_bounded_and_disjoint() -> None:
    candidates = [Path(f"c_{index:03d}") for index in range(500)]
    subsets = [
        [candidates[index], candidates[index + 1]]
        for index in range(0, 60, 2)
    ]
    # Add many overlapping alternatives to exercise the large-pool path.
    subsets.extend(
        [candidates[0], candidates[index]]
        for index in range(61, 500)
    )

    picked = _pick_candidate_disjoint_subsets(subsets, 30)

    assert len(picked) == 30
    flattened = [candidate for subset in picked for candidate in subset]
    assert len(flattened) == len(set(flattened))