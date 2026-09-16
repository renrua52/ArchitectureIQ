#!/usr/bin/env python3
"""Generate and learn fresh v1.5 KB epochs until interrupted."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPOCHS_DIR = ROOT / "data" / "kb_learning_v15_epochs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def next_epoch(kb_dir: Path) -> int:
    running: list[int] = []
    for path in (kb_dir / "runs").glob("epoch_*/manifest.json"):
        manifest = read_json(path)
        if manifest.get("status") == "running":
            running.append(int(manifest["epoch"]))
    if len(running) > 1:
        raise RuntimeError(f"Multiple running epochs found: {sorted(running)}")
    if running:
        return running[0]
    return int(read_json(kb_dir / "claims.json")["last_epoch"]) + 1


def run_checked(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--solver-model", default="claude-opus-5")
    parser.add_argument("--curator-model", default="gemini-3.6-flash")
    parser.add_argument("--solver-workers", type=int, default=6)
    parser.add_argument("--generation-workers", type=int, default=6)
    parser.add_argument("--remote-generation-host")
    parser.add_argument("--remote-generation-port", type=int, default=22)
    parser.add_argument("--remote-generation-root")
    parser.add_argument("--injection-limit", type=int, default=20)
    parser.add_argument("--curator-batch-size", type=int, default=16)
    parser.add_argument("--epoch-size", type=int, default=50)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--solver-max-tokens", type=int, default=16384)
    parser.add_argument("--curator-max-tokens", type=int, default=16384)
    parser.add_argument(
        "--max-epoch",
        type=int,
        help="Stop successfully after this epoch has completed (inclusive)",
    )
    args = parser.parse_args()

    if args.epoch_size != 50:
        raise ValueError("The v1.5 learning generator currently defines 50 questions per epoch")
    if not (args.kb_dir / "claims.json").exists():
        run_checked(
            [
                sys.executable,
                "tools/kb_pipeline.py",
                "init",
                "--kb-dir",
                str(args.kb_dir),
            ]
        )

    while True:
        epoch = next_epoch(args.kb_dir)
        if args.max_epoch is not None and epoch > args.max_epoch:
            print(f"Reached max epoch {args.max_epoch}; stopping.", flush=True)
            return 0
        epoch_root = DEFAULT_EPOCHS_DIR / f"epoch_{epoch:04d}"
        try:
            if args.remote_generation_host:
                if not args.remote_generation_root:
                    raise ValueError(
                        "--remote-generation-root is required with "
                        "--remote-generation-host"
                    )
                run_checked(
                    [
                        sys.executable,
                        "tools/run_remote_kb_generation.py",
                        "--host",
                        args.remote_generation_host,
                        "--port",
                        str(args.remote_generation_port),
                        "--remote-root",
                        args.remote_generation_root,
                        "--epoch",
                        str(epoch),
                        "--workers",
                        str(args.generation_workers),
                    ]
                )
            else:
                run_checked(
                    [
                        sys.executable,
                        "tools/build_kb_learning_v15.py",
                        "--epoch",
                        str(epoch),
                        "--workers",
                        str(args.generation_workers),
                        "--retry-failed",
                    ]
                )
            question_count = sum(
                1
                for path in epoch_root.iterdir()
                if path.is_dir()
                and (path / "question.json").is_file()
                and (path / "prompt.txt").is_file()
            )
            if question_count != args.epoch_size:
                raise RuntimeError(
                    f"Epoch {epoch} has {question_count}/{args.epoch_size} questions"
                )
            run_checked(
                [
                    sys.executable,
                    "tools/kb_pipeline.py",
                    "run-epoch",
                    "--kb-dir",
                    str(args.kb_dir),
                    "--questions-root",
                    str(epoch_root),
                    "--solver-model",
                    args.solver_model,
                    "--curator-model",
                    args.curator_model,
                    "--limit",
                    str(args.epoch_size),
                    "--injection-limit",
                    str(args.injection_limit),
                    "--curator-batch-size",
                    str(args.curator_batch_size),
                    "--workers",
                    str(args.solver_workers),
                    "--skip-seen",
                    "--no-show",
                    "--timeout",
                    str(args.timeout),
                    "--solver-max-tokens",
                    str(args.solver_max_tokens),
                    "--curator-max-tokens",
                    str(args.curator_max_tokens),
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"epoch={epoch} command failed with exit code {exc.returncode}; "
                f"retrying in {args.retry_delay:g}s",
                flush=True,
            )
            time.sleep(args.retry_delay)
        except RuntimeError as exc:
            print(f"epoch={epoch} incomplete: {exc}; retrying", flush=True)
            time.sleep(args.retry_delay)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
        raise SystemExit(130)
