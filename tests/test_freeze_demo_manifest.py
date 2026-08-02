"""Regression checks for portable demo-release freeze manifests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "freeze_demo_manifest.py"
SPEC = importlib.util.spec_from_file_location("freeze_demo_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
freeze_demo_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_demo_manifest)


def test_repo_relative_uses_portable_repository_paths() -> None:
    assert freeze_demo_manifest.repo_relative(ROOT / "profiles" / "v1.yaml") == "profiles/v1.yaml"


def test_repo_relative_rejects_paths_outside_repository() -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        freeze_demo_manifest.repo_relative(ROOT.parent / "outside.json")


def test_tracked_release_manifest_uses_repository_relative_paths() -> None:
    manifest_path = ROOT / "outputs" / "demo_release_integration" / "RELEASE_FREEZE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "demo_release_freeze_v2"
    assert manifest["path_base"] == "repository_root"

    paths = [
        manifest["collection_path"],
        *(item["path"] for item in manifest["profiles"].values()),
        manifest["frontend_bakefile"]["path"],
        manifest["release_spec"]["path"],
        manifest["external_data_bundle"]["path"],
    ]
    assert all(not Path(path).is_absolute() and "\\" not in path for path in paths)