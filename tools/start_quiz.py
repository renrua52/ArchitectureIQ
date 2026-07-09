#!/usr/bin/env python3
"""Idempotently launch the local ArchitectureIQ quiz inspector."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


DEFAULT_RUN = (
    "data/datasets/univariate_regression/sym_62678b/"
    "questions/run_20q_3c_b09206/q_79e34e"
)


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


def wait_until_running(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(port):
            return True
        time.sleep(0.25)
    return False


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
        default=DEFAULT_RUN,
        help="Question run or question directory to open first.",
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

    question_run = (root / args.question_run).resolve()
    if not question_run.exists():
        print(f"Question run not found: {question_run}", file=sys.stderr)
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
        "--",
        str(question_run),
    ]

    print(f"Starting ArchitectureIQ quiz on {url}")
    print("Press Ctrl-C in this terminal to stop it.")
    if not args.no_browser:
        webbrowser.open(url)

    process = subprocess.Popen(cmd, cwd=root)
    if wait_until_running(args.port, timeout=8):
        print(f"ArchitectureIQ quiz is ready: {url}")
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
