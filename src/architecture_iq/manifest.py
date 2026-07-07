from __future__ import annotations

import platform
import sys
from typing import Any

import torch

from architecture_iq.paths import ROOT
from architecture_iq.util import git_commit_hash, write_json


def write_benchmark_manifest(profile_name: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "profile": profile_name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_commit": git_commit_hash(ROOT),
    }
    if extra:
        payload.update(extra)
    write_json(ROOT / "benchmark_manifest.json", payload)
