#!/usr/bin/env python3
"""Run Gate 1/2 provenance and GT-health audits for candidate sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from architecture_iq.profile import load_profile
from question_audit_lib import audit_question_inputs, markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-set", action="append", type=Path, required=True)
    parser.add_argument("--profile", default="v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_question_inputs(args.dataset, list(args.candidate_set), load_profile(args.profile))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "question_input_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "question_input_audit.md").write_text(markdown_report("Question input audit (Gate 1/2)", report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
