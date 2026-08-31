"""The downloadable repro bundle: layout, gating, TS/Python parity, and a real run.

The bundle is built twice — in the browser by `frontend/quiz/src/bundle.ts` and on
the command line by `tools/export_repro_bundle.py`. These tests pin the contract
both must satisfy, and finish by actually running `reproduce.py` against recorded
ground truth.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAKE = ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"
QUIZ = ROOT / "frontend" / "quiz"

sys.path.insert(0, str(ROOT / "tools"))
from export_repro_bundle import build_bundle_entries, write_bundle  # noqa: E402

STATIC_FILES = ("README.md", "reproduce.py")
CHOICE_FILES = ("candidate_spec.json", "model.py", "loss.py", "optimizer.py", "train.py")


@pytest.fixture(scope="module")
def bake() -> dict:
    if not BAKE.exists():
        pytest.skip(f"no BakeFile at {BAKE}")
    with BAKE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pick(bake: dict, predicate=lambda question: True) -> dict:
    for summary in bake["questions"]:
        question = bake["byId"][summary["id"]]
        if predicate(question):
            return question
    pytest.skip("no matching question in the bake")


def _cheapest(bake: dict) -> dict:
    """Smallest total_samples_seen among MLP questions — the fastest thing to re-run."""
    def cost(question: dict) -> tuple[int, int]:
        specs = [choice["files"]["candidate_spec.json"] for choice in question["detail"]["choices"]]
        types = {spec["model"]["type"] for spec in specs}
        return (
            0 if types == {"mlp"} else 1,
            max(int(spec["budget"]["total_samples_seen"]) for spec in specs),
        )

    candidates = [bake["byId"][summary["id"]] for summary in bake["questions"]]
    return min(candidates, key=cost)


def test_answered_bundle_has_every_file(bake: dict) -> None:
    question = _pick(bake)
    entries = build_bundle_entries(question, True)
    root = question["id"]
    for name in (*STATIC_FILES, "question.json", "prompt.txt"):
        assert f"{root}/{name}" in entries
    assert f"{root}/dataset/synthesize.py" in entries
    assert f"{root}/dataset/dataset_spec.json" in entries
    for choice in question["detail"]["choices"]:
        for name in CHOICE_FILES:
            assert f"{root}/choices/{choice['letter']}/{name}" in entries
        assert f"{root}/choices/{choice['letter']}/reference/summary.json" in entries
    assert entries[f"{root}/prompt.txt"] == question["detail"]["prompt"]


def test_preanswer_bundle_withholds_the_answer(bake: dict) -> None:
    question = _pick(bake)
    entries = build_bundle_entries(question, False)
    assert not [path for path in entries if "/reference/" in path]
    meta = json.loads(entries[f"{question['id']}/question.json"])
    assert meta["answered"] is False
    assert "correct_letter" not in meta
    assert "ranked" not in meta
    # The code is identical; only the results are held back.
    answered = build_bundle_entries(question, True)
    for path, content in entries.items():
        if path.endswith("question.json"):
            continue
        assert answered[path] == content
    assert json.loads(answered[f"{question['id']}/question.json"])["correct_letter"] == (
        question["reveal"]["correctLetter"]
    )
    # No recorded score leaks in either, however it is spelled.
    blob = "".join(entries.values())
    for ranked in question["reveal"]["ranked"]:
        assert repr(ranked["mean"]) not in blob
        assert f"{ranked['mean']:.6f}" not in blob


def test_question_json_matches_the_specs(bake: dict) -> None:
    question = _pick(bake)
    meta = json.loads(build_bundle_entries(question, True)[f"{question['id']}/question.json"])
    assert meta["bundle_version"] == 1
    assert meta["selection_metric"] == question["detail"]["dataset"]["selectionMetric"]
    assert meta["n_seeds"] == question["evaluation"]["n_seeds"]
    assert meta["base_seed"] == question["evaluation"]["base_seed"]
    assert meta["fail_threshold"] == (
        question["detail"]["dataset"]["files"]["dataset_spec.json"]["significance"]["fail_threshold"]
    )
    for entry, choice in zip(meta["choices"], question["detail"]["choices"], strict=True):
        budget = choice["files"]["candidate_spec.json"]["budget"]
        assert entry["letter"] == choice["letter"]
        assert entry["candidate_id"] == choice["candidateId"]
        assert entry["training_steps"] == budget["training_steps"]
        assert entry["batch_size"] == budget["batch_size"]
        assert entry["total_samples_seen"] == budget["total_samples_seen"]


def test_written_zip_and_directory_agree(bake: dict, tmp_path: Path) -> None:
    import zipfile

    question = _pick(bake)
    entries = build_bundle_entries(question, True)
    archive = write_bundle(entries, tmp_path / "bundle.zip")
    tree = write_bundle(entries, tmp_path / "tree")
    with zipfile.ZipFile(archive) as handle:
        assert set(handle.namelist()) == set(entries)
        for path in entries:
            assert handle.read(path).decode("utf-8") == entries[path]
    for path, content in entries.items():
        assert (tree / path).read_text(encoding="utf-8") == content


@pytest.mark.parametrize("answered", [True, False])
def test_browser_and_cli_builders_agree(bake: dict, answered: bool) -> None:
    """The shipped TS builder and the Python exporter must not drift apart.

    Numbers re-serialized from the bake (`3e-05` vs `0.00003`) legitimately differ
    in spelling between the two JSON writers, so JSON files are compared parsed and
    everything else byte for byte.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    if not (QUIZ / "node_modules" / "esbuild").exists():
        pytest.skip("frontend dependencies are not installed")

    question = _pick(bake)
    command = ["node", "scripts/dump-bundle.mjs", "--bake", str(BAKE), "--question", question["id"]]
    if answered:
        command.append("--answered")
    process = subprocess.run(command, cwd=QUIZ, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    from_ts = json.loads(process.stdout)
    from_py = build_bundle_entries(question, answered)

    assert set(from_ts) == set(from_py)
    for path in from_py:
        if from_ts[path] == from_py[path]:
            continue
        assert path.endswith(".json"), f"{path} differs between bundle.ts and export_repro_bundle.py"
        assert json.loads(from_ts[path]) == json.loads(from_py[path]), path


def test_reproduce_py_reproduces_recorded_ground_truth(bake: dict, tmp_path: Path) -> None:
    """End-to-end: run the bundled reproduce.py and check it lands on the recorded value."""
    question = _cheapest(bake)
    root = question["id"]
    write_bundle(build_bundle_entries(question, True), tmp_path)
    bundle = tmp_path / root

    letter = question["reveal"]["correctLetter"]
    metric = question["detail"]["dataset"]["selectionMetric"]
    recorded = next(
        entry[f"final_{metric}"]
        for entry in question["reveal"]["files"][letter]["summary.json"]["seed_results"]
        if entry["seed"] == question["evaluation"]["base_seed"]
    )

    process = subprocess.run(
        [sys.executable, "reproduce.py", "--seeds", "1", "--letters", letter],
        cwd=bundle,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "within relative tolerance" in process.stdout, process.stdout

    line = next(line for line in process.stdout.splitlines() if f"{metric}=" in line)
    value = float(line.split(f"{metric}=")[1].split()[0])
    assert value == pytest.approx(recorded, rel=1e-3), process.stdout
