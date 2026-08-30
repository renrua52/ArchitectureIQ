#!/usr/bin/env python3
"""Re-derive a BakeFile's classification scatter payloads from materialized tensors.

`tools/export_quiz_static.py` is the only place that builds plot payloads, and a
full re-export is always preferable to this script. It exists for one narrow
case: a bake produced by a newer exporter than the checked-out branch can run
(e.g. model specs whose natural-language formatter lives elsewhere). Re-exporting
then would silently rewrite question content, so this rebuilds *only*
`detail.dataset.plot` for classification families, calling the exporter's own
`_classification_plot` on the dataset instance's own `train.pt` / `test.pt`.

Everything else in the bake is passed through untouched, and the script refuses
to run if the recomputed probability grid disagrees with the stored one — that
disagreement means the exporter really has moved on and a re-export is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "tools" / "question_inspector"
if str(INSPECTOR) not in sys.path:
    sys.path.insert(0, str(INSPECTOR))


def _load_exporter() -> Any:
    """Import export_quiz_static without triggering its CLI."""
    spec = importlib.util.spec_from_file_location(
        "export_quiz_static", ROOT / "tools" / "export_quiz_static.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tools/export_quiz_static.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keys that describe the projection rather than the sampled points. They must
# survive the rebuild unchanged; if they do not, this script is the wrong tool.
GRID_KEYS = (
    "kind",
    "xEdges",
    "yEdges",
    "probability",
    "featurePair",
    "selectionNote",
    "xLabel",
    "yLabel",
    "legend",
    "min",
    "max",
)


def _dataset_dir(data_root: Path, family: str, dataset_id: str) -> Path:
    return data_root / "datasets" / family / dataset_id


def refresh(bake: dict[str, Any], data_root: Path) -> list[str]:
    exporter = _load_exporter()
    import torch

    touched: list[str] = []
    for question_id, question in bake.get("byId", {}).items():
        dataset = question.get("detail", {}).get("dataset", {})
        plot = dataset.get("plot") or {}
        if plot.get("kind") != "classification":
            continue
        directory = _dataset_dir(data_root, dataset["family"], dataset["datasetId"])
        train_path = directory / "train.pt"
        test_path = directory / "test.pt"
        if not (train_path.is_file() and test_path.is_file()):
            raise SystemExit(f"{question_id}: missing materialized tensors under {directory}")
        params = json.loads((directory / "dataset_spec.json").read_text())["params"]
        rebuilt = exporter._classification_plot(
            torch.load(train_path, weights_only=True),
            torch.load(test_path, weights_only=True),
            params,
        )
        for key in GRID_KEYS:
            if json.dumps(rebuilt.get(key)) != json.dumps(plot.get(key)):
                raise SystemExit(
                    f"{question_id}: recomputed {key!r} differs from the bake — "
                    "the exporter has changed, re-export instead of refreshing"
                )
        plot["train"] = rebuilt["train"]
        plot["test"] = rebuilt["test"]
        touched.append(question_id)
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bake", type=Path, nargs="+", help="BakeFile(s) to rewrite in place")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    for path in args.bake:
        bake = json.loads(path.read_text())
        touched = refresh(bake, data_root)
        path.write_text(json.dumps(bake, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"refreshed {len(touched)} classification plots in {path}: {', '.join(touched)}")


if __name__ == "__main__":
    main()
