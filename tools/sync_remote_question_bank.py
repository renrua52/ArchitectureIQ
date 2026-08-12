#!/usr/bin/env python3
"""Sync the remote ArchitectureIQ V1 human pre-review bank into this repo.

Sources (either fetched from the live site or read from a local mirror):
  https://architecture-iq.com/data/index.json
  https://architecture-iq.com/data/by-id/<id>.json

Outputs:
  frontend/quiz/public/data/questions.json                          -> BakeFile
  benchmark_releases/question_packs/v1-human-250q-<hash>/           -> Inspector pack

Usage:
  python tools/sync_remote_question_bank.py [--index PATH|URL] [--by-id-dir DIR]
      [--no-materialize] [--out-pack DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from architecture_iq.runtime.loader import load_synthesize_module  # noqa: E402
from architecture_iq.util import short_hash, write_json  # noqa: E402

REMOTE_INDEX = "https://architecture-iq.com/data/index.json"
REMOTE_BY_ID = "https://architecture-iq.com/data/by-id/{qid}.json"
BAKE_OUT = ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"
PACKS_ROOT = ROOT / "benchmark_releases" / "question_packs"
RUN_ID = "run_v1_human_prereview"
SET_ID = "set_v1_human_prereview"
INSPECTOR_PROFILE = "v1_human"


def _read_json(path_or_url: str) -> dict[str, Any]:
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(path_or_url).read_text(encoding="utf-8"))


def _load_questions(index_path: str, by_id_dir: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    index = _read_json(index_path)
    rows = index["questions"]
    questions: dict[str, Any] = {}
    if by_id_dir:
        mirror = Path(by_id_dir)
        missing = [row["id"] for row in rows if not (mirror / f"{row['id']}.json").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} by-id files missing in {mirror}: {missing[:5]}...")
        for row in rows:
            qid = row["id"]
            questions[qid] = json.loads((mirror / f"{qid}.json").read_text(encoding="utf-8"))
    else:
        for row in rows:
            qid = row["id"]
            questions[qid] = _read_json(REMOTE_BY_ID.format(qid=qid))
    return index, questions


def build_bake(index: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    collection = dict(index.get("collection") or {})
    collection.setdefault("question_count", len(index["questions"]))
    collection.setdefault(
        "source",
        "https://architecture-iq.com (synced by tools/sync_remote_question_bank.py)",
    )
    bake: dict[str, Any] = {
        "schema_version": 1,
        "ordered": True,
        "collection": collection,
        "questions": index["questions"],
        "byId": {row["id"]: questions[row["id"]] for row in index["questions"]},
    }
    return bake


def _dataset_files(q: dict[str, Any]) -> dict[str, Any]:
    files = q["detail"]["dataset"].get("files") or {}
    spec = files.get("dataset_spec.json")
    synthesize = files.get("synthesize.py")
    if not isinstance(spec, dict) or not isinstance(synthesize, str):
        raise ValueError(f"{q['id']}: dataset files missing dataset_spec.json or synthesize.py")
    return {"spec": spec, "synthesize": synthesize}


def _materialize_dataset(q: dict[str, Any], ds_dir: Path) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    files = _dataset_files(q)
    write_json(ds_dir / "dataset_spec.json", files["spec"])
    (ds_dir / "synthesize.py").write_text(files["synthesize"], encoding="utf-8")
    module = load_synthesize_module(ds_dir / "synthesize.py")
    result = module.synthesize()
    if isinstance(result, dict):
        train_x, train_y = result["x"], result["y"]
        test_x, test_y = result["test_x"], result["test_y"]
    else:
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError(f"{q['id']}: synthesize() returned unexpected shape {type(result)}")
        train_x, train_y, test_x, test_y = result
    torch.save({"x": train_x, "y": train_y}, ds_dir / "train.pt")
    torch.save({"x": test_x, "y": test_y}, ds_dir / "test.pt")


def _materialize_candidate(q: dict[str, Any], choice: dict[str, Any], cand_dir: Path) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    files = choice.get("files") or {}
    for name in ("candidate_spec.json", "model.py", "loss.py", "optimizer.py", "train.py"):
        value = files.get(name)
        if value is None:
            raise ValueError(f"{q['id']}: choice {choice['letter']} missing {name}")
        if name.endswith(".json"):
            write_json(cand_dir / name, value)
        else:
            (cand_dir / name).write_text(value, encoding="utf-8")
    results_dir = cand_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    reveal = q.get("reveal") or {}
    choice_files = reveal.get("files") or {}
    summary = (choice_files.get(choice["letter"]) or {}).get("summary.json")
    if isinstance(summary, dict):
        write_json(results_dir / "summary.json", summary)
    curve = next(
        (entry for entry in reveal.get("curves") or [] if entry.get("letter") == choice["letter"]),
        None,
    )
    if curve is not None and curve.get("mean"):
        mean = np.asarray(curve["mean"], dtype=np.float64)
        samples = np.asarray(curve["samples"], dtype=np.int64)
        np.savez(results_dir / "curves.npz", curves=mean.reshape(1, -1), samples=samples)


def _materialize_question(
    q: dict[str, Any],
    q_rel: Path,
    ds_rel: Path,
    data_root: Path,
) -> None:
    qdir = data_root / q_rel
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "prompt.txt").write_text(q["detail"]["prompt"], encoding="utf-8")
    budget = q["budget"]
    total_samples = int(budget["total_samples_seen"]) if isinstance(budget, dict) else int(budget)
    first_spec = q["detail"]["choices"][0]["files"].get("candidate_spec.json") or {}
    question = {
        "schema_version": "1.0",
        "family": q["family"],
        "dataset_id": q["datasetId"],
        "budget": {"total_samples_seen": total_samples},
        "type": q["type"],
        "invariant_axes": list(q.get("invariantAxes") or []),
        "varying_axes": list(q.get("varyingAxes") or []),
        "num_choices": int(q.get("numChoices") or len(q["detail"]["choices"])),
        "choices": [
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidateId"],
                "candidate_path": str(ds_rel / "candidates" / SET_ID / choice["candidateId"]),
                "candidate_set_path": str(ds_rel / "candidates" / SET_ID),
            }
            for choice in q["detail"]["choices"]
        ],
        "correct_letter": (q.get("reveal") or {}).get("correctLetter", "A"),
        "evaluation": dict(q.get("evaluation") or {}),
        "prompt": {"template_version": "1.0", "rendered_path": "prompt.txt"},
        "question_id": q["id"],
        "profile": INSPECTOR_PROFILE,
        "profile_hash": first_spec.get("profile_hash"),
        "track": q.get("track"),
        "llm_difficulty": q.get("llmDifficulty"),
        "dataset_bucket": q.get("datasetBucket"),
        "question_run_id": RUN_ID,
        "question_run_path": str(q_rel.parent),
    }
    write_json(qdir / "question.json", question)


def materialize_pack(index: dict[str, Any], questions: dict[str, Any], out_pack: Path) -> Path:
    pack_id = f"v1-human-250q-{short_hash(index['questions'])[:8]}"
    pack_root = out_pack / pack_id
    data_root = pack_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    written_datasets: set[str] = set()
    written_candidates: set[str] = set()
    question_paths: list[str] = []
    for row in index["questions"]:
        q = questions[row["id"]]
        family = q["family"]
        ds_id = q["datasetId"]
        ds_rel = Path("datasets") / family / ds_id
        ds_dir = data_root / ds_rel
        if ds_id not in written_datasets:
            _materialize_dataset(q, ds_dir)
            written_datasets.add(ds_id)
        for choice in q["detail"]["choices"]:
            cid = choice["candidateId"]
            if cid not in written_candidates:
                _materialize_candidate(q, choice, data_root / ds_rel / "candidates" / SET_ID / cid)
                written_candidates.add(cid)
        q_rel = ds_rel / "questions" / RUN_ID / q["id"]
        _materialize_question(q, q_rel, ds_rel, data_root)
        question_paths.append(str(q_rel))

    collection = {
        "schema_version": "question_review_collection_v1",
        "collection_id": f"v1_human_prereview_250q_{short_hash(index['questions'])[:8]}",
        "title": "ArchitectureIQ V1 · human pre-review (250 questions)",
        "question_paths": question_paths,
        "source_runs": ["v1_human_prereview_20260804"],
        "candidate_reuse_policy": "benchmark_v1_llm",
        "profiles": [INSPECTOR_PROFILE],
    }
    write_json(pack_root / "collection.json", collection)
    write_json(
        pack_root / "pack.json",
        {
            "schema_version": "question_pack_v1",
            "pack_id": pack_id,
            "display_name": "V1 human pre-review · 250-question pack",
            "question_count": len(index["questions"]),
            "collection_path": "collection.json",
            "data_root": "data",
            "profile": INSPECTOR_PROFILE,
            "provenance": {
                "collection_id": collection["collection_id"],
                "remote_index": "https://architecture-iq.com/data/index.json",
                "note": (
                    "Synced by tools/sync_remote_question_bank.py; "
                    "profiles/v1_human.yaml mirrors the remote generation pool."
                ),
            },
        },
    )
    return pack_root


def validate_bake(path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import validate_quiz_bake  # type: ignore[import-not-found]

    schema = json.loads(
        (ROOT / "contracts" / "quiz_bake.schema.json").read_text(encoding="utf-8")
    )
    errors = validate_quiz_bake.validate_file(path, schema)
    if errors:
        raise SystemExit("\n".join(f"bake error: {item}" for item in errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=REMOTE_INDEX, help="index.json path or URL")
    parser.add_argument("--by-id-dir", default=None, help="directory of pre-fetched by-id JSON files")
    parser.add_argument("--no-materialize", action="store_true", help="only rebuild the frontend bake")
    parser.add_argument(
        "--out-pack",
        default=PACKS_ROOT,
        help="question-pack root (default: benchmark_releases/question_packs)",
    )
    args = parser.parse_args()

    index, questions = _load_questions(args.index, args.by_id_dir)
    bake = build_bake(index, questions)
    BAKE_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(BAKE_OUT, bake)
    print(f"wrote bake: {BAKE_OUT} ({len(index['questions'])} questions)")
    validate_bake(BAKE_OUT)
    print("bake validated against contracts/quiz_bake.schema.json")

    if not args.no_materialize:
        pack_root = materialize_pack(index, questions, Path(args.out_pack))
        print(f"wrote inspector pack: {pack_root}")


if __name__ == "__main__":
    main()
