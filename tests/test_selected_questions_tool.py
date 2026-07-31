from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type


REPO = Path(__file__).resolve().parents[1]


def _load_tool():
    path = REPO / "tools" / "batch_generate" / "selected_questions.py"
    spec = importlib.util.spec_from_file_location("selected_questions_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SELECTED = _load_tool()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary(candidate_id: str, mean: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "selection_metric": "test_mse",
        "n_seeds": 10,
        "base_seed": 0,
        "excluded": False,
        "failed_seeds": 0,
        "mean_test_mse": mean,
        "std_test_mse": 0.001,
        "seed_results": [
            {
                "seed": seed,
                "failed": False,
                "final_test_mse": mean,
            }
            for seed in range(10)
        ],
    }


def _patch_data_root(monkeypatch: pytest.MonkeyPatch, data_root: Path) -> None:
    import architecture_iq.paths as paths_module
    import architecture_iq.prompts.renderer as renderer_module
    import architecture_iq.questions.generator as generator_module
    import architecture_iq.questions.runs as runs_module

    monkeypatch.setattr(paths_module, "DATA_DIR", data_root)
    monkeypatch.setattr(renderer_module, "DATA_DIR", data_root)
    monkeypatch.setattr(generator_module, "DATA_DIR", data_root)
    monkeypatch.setattr(runs_module, "DATA_DIR", data_root)
    monkeypatch.setattr(SELECTED, "DATA_DIR", data_root)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any], list[Path]]:
    ensure_registries()
    profile = load_profile("v1")
    data_root = tmp_path / "data"
    _patch_data_root(monkeypatch, data_root)

    dataset = (
        data_root / "datasets" / "univariate_regression" / "sym_selected_test"
    )
    dataset.mkdir(parents=True)
    _write_json(
        dataset / "dataset_spec.json",
        {
            "schema_version": "1.0",
            "dataset_id": "sym_selected_test",
            "family": "univariate_regression",
            "selection_metric": "test_mse",
            "params": {
                "expression": "x",
                "domain": [0.0, 1.0],
                "train_size": 8,
                "test_size": 8,
                "point_sampling": {"seed": 11},
            },
        },
    )
    (dataset / "synthesize.py").write_text(
        """import torch

def target(x: torch.Tensor) -> torch.Tensor:
    return x

def synthesize():
    return None
""",
        encoding="utf-8",
    )

    candidate_set = dataset / "candidates" / "set_selected_test"
    candidate_set.mkdir(parents=True)
    _write_json(
        candidate_set / "set.json",
        {
            "schema_version": "1.0",
            "set_id": candidate_set.name,
            "dataset_id": "sym_selected_test",
            "family": "univariate_regression",
            "profile": "v1",
        },
    )

    candidate_paths: list[Path] = []
    for index, (width, mean) in enumerate(
        ((16, 0.10), (32, 0.25), (64, 0.40), (128, 0.55))
    ):
        model = {
            "type": "mlp",
            "input_dim": 1,
            "depth": 1,
            "width": width,
            "residual": False,
            "layer_norm": [False],
            "activations": ["relu" if index % 2 == 0 else "gelu"],
        }
        candidate_spec = build_candidate_spec(
            profile,
            dataset_id="sym_selected_test",
            family="univariate_regression",
            budget=1024,
            batch_size=16,
            model=model,
            optimizer={
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            loss={"loss_id": "mse"},
        )
        candidate_path = candidate_set / candidate_spec["candidate_id"]
        write_candidate(candidate_spec, candidate_path, get_model_type("mlp"))
        _write_json(
            candidate_path / "results" / "summary.json",
            _summary(candidate_spec["candidate_id"], mean),
        )
        candidate_paths.append(candidate_path.resolve())

    plan = {
        "profile": "v1",
        "dataset_path": str(dataset),
        "candidate_set_paths": [str(candidate_set)],
        "seed": 12345,
        "questions": [
            {
                "candidate_ids": [
                    candidate_paths[0].name,
                    candidate_paths[1].name,
                    candidate_paths[2].name,
                ]
            },
            {
                "candidate_paths": [
                    str(candidate_paths[0]),
                    str(candidate_paths[2]),
                    str(candidate_paths[3]),
                ]
            },
        ],
    }
    return profile, plan, candidate_paths


def test_builds_canonical_run_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, plan, _ = _fixture(tmp_path, monkeypatch)

    with (
        patch.object(
            SELECTED,
            "build_question_record",
            wraps=SELECTED.build_question_record,
        ) as build_record,
        patch.object(
            SELECTED,
            "write_run_manifest",
            wraps=SELECTED.write_run_manifest,
        ) as write_manifest,
        patch.object(
            SELECTED,
            "write_prompt",
            wraps=SELECTED.write_prompt,
        ) as render_prompt,
        patch(
            "architecture_iq.ground_truth.runner.run_ground_truth",
            side_effect=AssertionError("selected question assembly must not train"),
        ),
    ):
        run_path, results = SELECTED.build_selected_question_run(
            plan,
            profile,
            base_dir=tmp_path,
        )

    assert build_record.call_count == 2
    assert write_manifest.call_count == 1
    assert render_prompt.call_count == 2
    manifest = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert manifest["num_questions"] == 2
    assert manifest["num_choices"] == 3
    assert manifest["seed"] == 12345
    assert manifest["question_ids"] == [record["question_id"] for record, _ in results]

    for record, question_path in results:
        assert (question_path / "question.json").is_file()
        prompt = (question_path / "prompt.txt").read_text(encoding="utf-8")
        assert "class Model(nn.Module):" in prompt
        assert "def loss_fn" in prompt
        assert "mean_test_mse" not in prompt
        assert record["significance"]["passed"] is True
        assert all(choice["candidate_path"].startswith("datasets/") for choice in record["choices"])

    with pytest.raises(FileExistsError):
        SELECTED.build_selected_question_run(plan, profile, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("summary_update", "expected"),
    [
        ({"excluded": True}, "excluded=True"),
        ({"failed_seeds": 1}, "failed_seeds=1"),
    ],
)
def test_rejects_ineligible_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_update: dict[str, Any],
    expected: str,
) -> None:
    profile, plan, candidate_paths = _fixture(tmp_path, monkeypatch)
    summary_path = candidate_paths[0] / "results" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(summary_update)
    _write_json(summary_path, summary)
    plan["questions"] = [plan["questions"][0]]

    with pytest.raises(ValueError, match=expected):
        SELECTED.build_selected_question_run(plan, profile, base_dir=tmp_path)


def test_rejects_candidate_outside_selected_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, plan, candidate_paths = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / candidate_paths[0].name
    plan["questions"] = [
        {
            "candidate_paths": [
                str(outside),
                str(candidate_paths[1]),
            ]
        }
    ]

    with pytest.raises(ValueError, match="not in the specified candidate sets"):
        SELECTED.build_selected_question_run(plan, profile, base_dir=tmp_path)


def test_canonical_record_builder_rejects_insignificant_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, plan, candidate_paths = _fixture(tmp_path, monkeypatch)
    summary_path = candidate_paths[1] / "results" / "summary.json"
    summary = _summary(candidate_paths[1].name, 0.12)
    _write_json(summary_path, summary)
    plan["questions"] = [plan["questions"][0]]

    with pytest.raises(ValueError, match="Significance failed"):
        SELECTED.build_selected_question_run(plan, profile, base_dir=tmp_path)

    dataset_path = Path(plan["dataset_path"])
    assert not (dataset_path / "questions").exists()
