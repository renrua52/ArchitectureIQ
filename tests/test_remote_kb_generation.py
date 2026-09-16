"""Tests for the remote KB question-generation adapter."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_remote_kb_generation import ssh_command, validate_remote_root  # noqa: E402


def test_validate_remote_root_requires_dedicated_cephfs_directory() -> None:
    assert validate_remote_root(
        "/cephfs/renzirui/architectureiq_kb_learning"
    ) == PurePosixPath("/cephfs/renzirui/architectureiq_kb_learning")

    for invalid in (
        "/cephfs/renzirui",
        "/cephfs/renzirui/../other",
        "/tmp/architectureiq_kb_learning",
        "relative/path",
    ):
        with pytest.raises(ValueError):
            validate_remote_root(invalid)


def test_ssh_command_does_not_write_known_hosts() -> None:
    command = ssh_command("root@example", 32151, ["mkdir", "-p", "/safe path"])

    assert command[:7] == [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        "32151",
    ]
    assert command[7] == "root@example"
    assert command[8] == "mkdir -p '/safe path'"
