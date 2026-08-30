#!/usr/bin/env python3
"""Re-derive a BakeFile's spec field rows from each choice's embedded spec.

`tools/export_quiz_static.py` is the only place that flattens specs into card
rows, and a full re-export is always preferable to this script. It exists for the
same narrow case as `tools/refresh_bake_plots.py`: a bake produced against specs
this branch's exporter can no longer re-bake. Re-exporting then would rewrite
question content, so this rebuilds *only* `detail.shared` and each
`detail.choices[].variant`, from the `candidate_spec.json` the bake already
carries, through the exporter's own `_shared_and_variant`.

Because the spec is embedded in the bake, nothing is read from `data/`: the rows
are recomputed from exactly the spec the question displays. The script refuses to
run if any label/value pair already in the bake would disappear — losing a row
means the exporter has genuinely moved on and a re-export is required. Gaining
rows is the point (a varying axis the old flattening dropped, e.g. `d_ff`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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


def _spec_of(choice: dict[str, Any]) -> dict[str, Any]:
    """The choice's candidate spec, whether the bake stored it parsed or as text."""
    raw = choice.get("files", {}).get("candidate_spec.json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise SystemExit(f"choice {choice.get('letter')!r} has no candidate_spec.json")


def _rows(fields: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(field["label"], field["value"]) for field in fields}


def refresh(bake: dict[str, Any]) -> dict[str, int]:
    exporter = _load_exporter()
    added: dict[str, int] = {}
    for question_id, question in sorted(bake.get("byId", {}).items()):
        detail = question.get("detail", {})
        choices = detail.get("choices", [])
        if not choices:
            continue
        specs = {choice["letter"]: _spec_of(choice) for choice in choices}
        shared, variant = exporter._shared_and_variant(specs)
        # Compare per letter: a field may legitimately move between the shared
        # panel and a card, so only the union each choice displays must not shrink.
        gained = 0
        for choice in choices:
            letter = choice["letter"]
            before = _rows(detail.get("shared", [])) | _rows(choice.get("variant", []))
            after = _rows(shared) | _rows(variant.get(letter, []))
            missing = before - after
            if missing:
                raise SystemExit(
                    f"{question_id} choice {letter}: recomputed rows drop "
                    f"{sorted(missing)} — the exporter has changed, re-export instead"
                )
            gained += len(after - before)
        detail["shared"] = shared
        for choice in choices:
            choice["variant"] = variant.get(choice["letter"], [])
        added[question_id] = gained
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bake", type=Path, nargs="+", help="BakeFile(s) to rewrite in place")
    args = parser.parse_args()
    for path in args.bake:
        bake = json.loads(path.read_text())
        added = refresh(bake)
        path.write_text(json.dumps(bake, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary = ", ".join(f"{qid}:+{count}" for qid, count in added.items())
        print(f"refreshed spec rows in {path} ({summary})")


if __name__ == "__main__":
    main()
