from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.meta_model_study import bounds


def _row(loss: float, *, std: float = 0.0, n_seeds: int = 4) -> dict:
    return {"target": {"mean_loss": loss, "std_loss": std, "n_seeds": n_seeds}}


def test_environment_baselines_account_for_ties_and_stability() -> None:
    result = bounds.analyze_environment([_row(1.0), _row(1.0), _row(2.0), _row(4.0)])

    assert result["observed_gt_oracle"] == {"top1_accuracy": 1.0, "mean_regret": 0.0}
    assert result["uniform_random_three_choice"]["n_triples"] == 4
    assert result["uniform_random_three_choice"]["top1_accuracy"] == pytest.approx(0.5)
    assert result["uniform_random_three_choice"]["mean_regret"] == pytest.approx(1.0)
    assert result["random_pair"] == {
        "n_comparable_pairs": 5,
        "pair_concordance": 0.5,
        "true_ties_excluded": 1,
    }
    proxy = result["winner_stability_empirical_proxy"]
    assert proxy["n_eligible_triples"] == 4
    assert proxy["n_stable_triples"] == 2
    assert proxy["stable_fraction"] == 0.5
    assert "not a theoretical ceiling" in proxy["label"]
    assert "not independent" in result["finite_sample"]["warning"]


def test_stability_proxy_is_conservative_and_optional() -> None:
    result = bounds.analyze_environment(
        [_row(1.0, std=1.0), _row(1.1, std=1.0), {"target": {"mean_loss": 2.0}}]
    )
    proxy = result["winner_stability_empirical_proxy"]
    assert proxy["n_eligible_triples"] == 0
    assert proxy["stable_fraction"] is None


def test_snapshot_cli_emits_per_environment_and_aggregate_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class Environment:
        experiment_id = "env_a"

        def rows(self, split: str) -> tuple[dict, ...]:
            assert split == "validation"
            return (_row(1.0), _row(2.0), _row(3.0))

    snapshot_path = tmp_path / "snapshot.json"
    fake = SimpleNamespace(
        corpus=SimpleNamespace(environments=(Environment(),)),
        path=snapshot_path.resolve(),
        sha256="a" * 64,
    )
    monkeypatch.setattr(bounds, "load_snapshot", lambda path: fake)

    assert bounds.main(["--snapshot-manifest", str(snapshot_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["environments"]["env_a"]["uniform_random_three_choice"][
        "top1_accuracy"
    ] == pytest.approx(1 / 3)
    assert output["aggregate"]["observed_gt_oracle"]["top1_accuracy"] == 1.0
    assert output["aggregate"]["uniform_random_three_choice"]["mean_regret"] is None
    assert "not pooled" in output["aggregate"]["uniform_random_three_choice"][
        "mean_regret_aggregation"
    ]
    assert output["aggregate"]["finite_sample"]["effective_n_environments"] == 1
    assert output["source"]["kind"] == "snapshot_manifest"
