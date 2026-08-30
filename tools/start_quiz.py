#!/usr/bin/env python3
"""Idempotently launch the local ArchitectureIQ quiz inspector."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/_stcore/health"


def quiz_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(health_url(port), timeout=1.5) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def resolve_question_run(root: Path, requested: str | None) -> Path | None:
    """Resolve the question to open, or None to let the inspector pick one.

    There is no bundled demo to fall back on: questions are generated artifacts
    under gitignored data/, so with nothing requested the inspector opens the
    first question it discovers there.
    """
    if requested is None:
        return None
    question_run = (root / requested).resolve()
    if not question_run.exists():
        raise FileNotFoundError(question_run)
    return question_run


def wait_until_running(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(port):
            return True
        time.sleep(0.25)
    return False


def wait_for_process(process: subprocess.Popen[bytes]) -> int:
    """Wait for Streamlit and treat Ctrl-C as a clean local shutdown."""
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local Streamlit port to use.",
    )
    parser.add_argument(
        "--question-run",
        default=None,
        help=(
            "Question, run, or review-collection JSON to open "
            "(default: the first question found under data/)."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start or reuse the service without opening a browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    url = quiz_url(args.port)

    if is_running(args.port):
        print(f"ArchitectureIQ quiz is already running: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    try:
        question_run = resolve_question_run(root, args.question_run)
    except FileNotFoundError as exc:
        print(f"Question run not found: {exc}", file=sys.stderr)
        return 1

    app = root / "tools" / "question_inspector" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.port),
    ]
    if question_run is not None:
        cmd += ["--", str(question_run)]

    print(f"Starting ArchitectureIQ quiz on {url}")
    print("Press Ctrl-C in this terminal to stop it.")
    if not args.no_browser:
        webbrowser.open(url)

    env = os.environ.copy()
    local_src = str(root / "src")
    env["PYTHONPATH"] = local_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.Popen(cmd, cwd=root, env=env)
    if wait_until_running(args.port, timeout=8):
        print(f"ArchitectureIQ quiz is ready: {url}")
    return wait_for_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
