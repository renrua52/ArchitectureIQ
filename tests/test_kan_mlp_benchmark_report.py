from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kan_mlp_benchmark_report", REPO / "tools" / "kan_mlp_benchmark_report.py"
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _row(label: str, metric: float, params: int, runtime: float, *, dataset: str = "d1", budget: int = 100) -> dict:
    return {
        "dataset_id": dataset,
        "budget": budget,
        "pair": "pair1",
        "label": label,
        "parameters": params,
        "elapsed_seconds": runtime,
        "seeds": 4,
        "failed_seeds": 1 if label == "kan" else 0,
        "mean_test_mse": metric,
    }


def test_summarise_reports_costs_failures_and_baselines() -> None:
    report = REPORT.summarise(
        [
            REPORT.normalise_row(_row("kan", 0.2, 120, 3.0)),
            REPORT.normalise_row(_row("mlp", 0.4, 100, 1.0)),
        ]
    )

    assert report["n_pairs"] == 1
    assert report["models"]["kan"]["parameters"]["mean"] == 120
    assert report["models"]["kan"]["failure_rate"] == 0.25
    assert report["models"]["mlp"]["runtime_seconds"]["mean"] == 1.0
    assert report["baselines"]["always_kan"]["accuracy"] == 1.0
    assert report["baselines"]["always_mlp"]["accuracy"] == 0.0
    assert report["baselines"]["more_params"]["accuracy"] == 1.0
    assert report["baselines"]["fewer_params"]["accuracy"] == 0.0
    assert report["baselines"]["faster_runtime"]["accuracy"] == 0.0
    assert report["baselines"]["lookup"]["accuracy"] == 1.0


def test_load_rows_accepts_rows_envelope_and_infers_parameters(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "dataset_id": "d",
                        "budget": 8,
                        "pair": "p",
                        "label": "kan",
                        "model": {
                            "type": "kan",
                            "input_dim": 1,
                            "output_dim": 1,
                            "depth": 1,
                            "width": 2,
                            "grid_size": 3,
                            "spline_order": 3,
                        },
                        "seed_results": [{"failed": True}, {"failed": False}],
                        "mean_test_mse": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = REPORT.load_rows(path)
    assert rows[0]["parameters"] == (1 * 2 + 2 * 2 + 2 * 1) * 7
    assert rows[0]["failed_seeds"] == 1
    assert rows[0]["failure_rate"] == 0.5


def test_render_markdown_contains_baseline_table() -> None:
    report = REPORT.summarise(
        [REPORT.normalise_row(_row("kan", 0.2, 120, 3.0)), REPORT.normalise_row(_row("mlp", 0.4, 100, 1.0))]
    )
    markdown = REPORT.render_markdown(report)
    assert "## Baselines" in markdown
    assert "always_kan" in markdown
