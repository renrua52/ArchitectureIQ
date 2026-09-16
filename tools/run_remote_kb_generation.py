#!/usr/bin/env python3
"""Run one KB-learning generation epoch remotely and fetch its questions."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
LOCAL_EPOCHS_DIR = ROOT / "data" / "kb_learning_v15_epochs"
ALLOWED_REMOTE_ROOT = PurePosixPath("/cephfs/renzirui")
SYNC_PATHS = (
    "pyproject.toml",
    "profiles",
    "src",
    "tools/build_benchmark_v15.py",
    "tools/build_kb_learning_v15.py",
)


def validate_remote_root(value: str) -> PurePosixPath:
    root = PurePosixPath(value)
    if ".." in root.parts or not root.is_absolute() or root == ALLOWED_REMOTE_ROOT:
        raise ValueError(
            "remote root must be a dedicated directory below /cephfs/renzirui"
        )
    try:
        root.relative_to(ALLOWED_REMOTE_ROOT)
    except ValueError as exc:
        raise ValueError("remote root must stay below /cephfs/renzirui") from exc
    return root


def ssh_command(host: str, port: int, command: list[str]) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(port),
        host,
        shlex.join(command),
    ]


def sync_code(host: str, port: int, remote_repo: PurePosixPath) -> None:
    subprocess.run(
        ssh_command(host, port, ["mkdir", "-p", str(remote_repo)]),
        cwd=ROOT,
        check=True,
    )
    pack_env = os.environ.copy()
    pack_env["COPYFILE_DISABLE"] = "1"
    pack = subprocess.Popen(
        ["tar", "-czf", "-", *SYNC_PATHS],
        cwd=ROOT,
        env=pack_env,
        stdout=subprocess.PIPE,
    )
    assert pack.stdout is not None
    unpack = subprocess.run(
        ssh_command(
            host,
            port,
            ["tar", "-xzf", "-", "-C", str(remote_repo)],
        ),
        cwd=ROOT,
        stdin=pack.stdout,
        check=False,
    )
    pack.stdout.close()
    pack_status = pack.wait()
    if pack_status != 0 or unpack.returncode != 0:
        raise subprocess.CalledProcessError(
            pack_status or unpack.returncode,
            "remote code sync",
        )


def run_remote_epoch(
    host: str,
    port: int,
    remote_root: PurePosixPath,
    epoch: int,
    workers: int,
) -> None:
    remote_repo = remote_root / "repo"
    torch_packages = remote_root / "torch_packages"
    python_packages = remote_root / "python_packages"
    python_path = f"{torch_packages}:{python_packages}:{remote_repo / 'src'}"
    subprocess.run(
        ssh_command(
            host,
            port,
            [
                "env",
                f"PYTHONPATH={python_path}",
                "python3",
                str(remote_repo / "tools" / "build_kb_learning_v15.py"),
                "--epoch",
                str(epoch),
                "--workers",
                str(workers),
                "--retry-failed",
            ],
        ),
        cwd=ROOT,
        check=True,
    )


def fetch_epoch(
    host: str,
    port: int,
    remote_root: PurePosixPath,
    epoch: int,
) -> None:
    epoch_name = f"epoch_{epoch:04d}"
    remote_epochs = remote_root / "repo" / "data" / "kb_learning_v15_epochs"
    LOCAL_EPOCHS_DIR.mkdir(parents=True, exist_ok=True)
    pack = subprocess.Popen(
        ssh_command(
            host,
            port,
            ["tar", "-chf", "-", "-C", str(remote_epochs), epoch_name],
        ),
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    assert pack.stdout is not None
    unpack = subprocess.run(
        ["tar", "-xf", "-", "-C", str(LOCAL_EPOCHS_DIR)],
        cwd=ROOT,
        stdin=pack.stdout,
        check=False,
    )
    pack.stdout.close()
    pack_status = pack.wait()
    if pack_status != 0 or unpack.returncode != 0:
        raise subprocess.CalledProcessError(
            pack_status or unpack.returncode,
            "remote epoch fetch",
        )

    epoch_dir = LOCAL_EPOCHS_DIR / epoch_name
    question_count = sum(
        1
        for path in epoch_dir.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and (path / "question.json").is_file()
        and (path / "prompt.txt").is_file()
    )
    if question_count != 50:
        raise RuntimeError(
            f"Fetched epoch {epoch} has {question_count}/50 materialized questions"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    remote_root = validate_remote_root(args.remote_root)
    remote_repo = remote_root / "repo"
    sync_code(args.host, args.port, remote_repo)
    run_remote_epoch(args.host, args.port, remote_root, args.epoch, args.workers)
    fetch_epoch(args.host, args.port, remote_root, args.epoch)
    print(f"Fetched epoch {args.epoch} with 50 materialized questions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
