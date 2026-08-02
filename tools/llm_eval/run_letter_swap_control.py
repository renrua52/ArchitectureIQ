#!/usr/bin/env python3
"""Letter-swap control experiment for the high_budget 4-choice release.

Runs the exact 13 zero-shot prompts used in ``vapi_*_high_budget13_zero_shot``
plus an A<->B content-swapped variant, and compares:
  * accuracy on original vs swapped answer keys,
  * letter distribution (position/letter bias check),
  * candidate-level agreement between original and swapped answers
    (content-driven models keep the same candidate; biased models do not).

Usage:
    python tools/llm_eval/run_letter_swap_control.py [--models claude-sonnet-5,gpt-5.6-terra]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "llm_eval"))

from llm_client import LLMClient, ModelConfig  # noqa: E402
from response_parser import parse_choice_letter, split_chain_of_thought  # noqa: E402

SOURCE_RUN = REPO / "llm_runs" / "vapi_claude_sonnet5_high_budget13_zero_shot"
OUT_ROOT = REPO / "llm_runs" / f"control_letterswap_{datetime.now().strftime('%Y%m%d')}"

SECTION_RE = re.compile(r"(### Choice [ABCD]\n\n)(.*?)(?=\n### Choice [ABCD]\n\n|## Your answer)", re.S)


def swap_ab_sections(prompt: str) -> str:
    """Swap the bodies of Choice A and Choice B, keeping letter labels in place."""
    sections = {m.group(1): m.group(2) for m in SECTION_RE.finditer(prompt)}
    if not {"### Choice A\n\n", "### Choice B\n\n"} <= set(sections):
        raise ValueError("prompt does not contain Choice A/B sections")

    def repl(m: re.Match) -> str:
        header = m.group(1)
        if header == "### Choice A\n\n":
            return header + sections["### Choice B\n\n"]
        if header == "### Choice B\n\n":
            return header + sections["### Choice A\n\n"]
        return header + m.group(2)

    new_prompt, n = SECTION_RE.subn(repl, prompt)
    if n != 4:
        raise ValueError(f"expected 4 choice sections, got {n}")
    return new_prompt


def swap_letter(letter: str) -> str:
    return {"A": "B", "B": "A"}.get(letter, letter)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="claude-sonnet-5,gpt-5.6-terra")
    ap.add_argument("--workers", type=int, default=13)
    args = ap.parse_args()

    relay = json.load(open(os.path.expanduser("~/.agents/relay.json")))
    base_url = relay["eval"]["base_url"].rstrip("/") + "/v1"
    api_key = relay["eval"]["api_key"]

    # Load the 13 source prompts + GT + candidate maps.
    rows = []
    for p in sorted((SOURCE_RUN / "results").glob("q_*.json")):
        r = json.load(open(p))
        rows.append(r)
    rows.sort(key=lambda r: r["question_id"])
    print(f"loaded {len(rows)} source questions from {SOURCE_RUN.name}")

    # candidate_id per letter per question (release question.json)
    qmap: dict[str, dict[str, str]] = {}
    for qp in (REPO / "data/releases/high_budget_confirmed_v1").rglob("question.json"):
        q = json.load(open(qp))
        if q.get("question_id"):
            qmap[q["question_id"]] = {c["letter"]: c["candidate_id"] for c in q["choices"]}

    client = LLMClient(base_url=base_url, api_key=api_key, timeout_s=300, max_retries=4)
    valid = frozenset("ABCD")

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_dir = OUT_ROOT / model.replace("/", "_")
        cfg = ModelConfig(name=model, temperature=0.0, max_tokens=65536)
        print(f"\n== model {model} -> {run_dir.relative_to(REPO)} ==")

        items = []
        for r in rows:
            items.append(
                {
                    "question_id": r["question_id"],
                    "gt": r["ground_truth_letter"],
                    "variant": "original",
                    "prompt": r["eval_prompt"],
                }
            )
            items.append(
                {
                    "question_id": r["question_id"],
                    "gt": swap_letter(r["ground_truth_letter"]),
                    "variant": "swapped_ab",
                    "prompt": swap_ab_sections(r["eval_prompt"]),
                }
            )

        def run_one(it: dict) -> dict:
            from completion import fetch_model_response
            exchange = fetch_model_response(
                client, it["prompt"], cfg, valid, max_continuations=2
            )
            parsed = parse_choice_letter(exchange.model_response, valid)
            return {
                **it,
                "parsed_letter": parsed,
                "correct": parsed == it["gt"] if parsed else False,
                "model_response": exchange.model_response,
                "finish_reason": exchange.finish_reason,
                "truncated": exchange.truncated,
                "usage": exchange.usage,
                "message_parts": exchange.message_parts,
            }

        results = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, it): it["question_id"] + "/" + it["variant"] for it in items}
            for fut in as_completed(futs):
                label = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {label}: {exc}")
                    continue
                results[label] = res
                print(f"  {label}: GT={res['gt']} ans={res['parsed_letter']} "
                      f"{'OK' if res['correct'] else '--'} ({len(res['model_response'])} chars)")

        # Persist raw records (per-question JSONL + prompt copy), keep every thinking process.
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
        for label, res in sorted(results.items()):
            qid, variant = label.split("/")
            (run_dir / "prompts" / f"{qid}_{variant}.txt").write_text(res["prompt"], encoding="utf-8")
            rec = {k: v for k, v in res.items() if k != "prompt"}
            with open(run_dir / "results" / f"{qid}_{variant}.json", "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)

        # Summaries per variant.
        summary = {}
        for variant in ("original", "swapped_ab"):
            vres = [results[k] for k in results if k.endswith("/" + variant)]
            parsed = [r for r in vres if r["parsed_letter"]]
            correct = [r for r in parsed if r["correct"]]
            dist = {}
            for r in parsed:
                dist[r["parsed_letter"]] = dist.get(r["parsed_letter"], 0) + 1
            summary[variant] = {
                "n": len(vres),
                "parsed": len(parsed),
                "correct": len(correct),
                "accuracy": len(correct) / len(parsed) if parsed else None,
                "letter_distribution": dist,
            }

        # Candidate-level agreement across variants (content-driven -> high).
        agreement = {"n": 0, "same_candidate": 0}
        per_q = {}
        for r in rows:
            qid = r["question_id"]
            a = results.get(f"{qid}/original")
            b = results.get(f"{qid}/swapped_ab")
            if not a or not b:
                continue
            mapping = qmap.get(qid)
            if not mapping or a["parsed_letter"] is None or b["parsed_letter"] is None:
                continue
            ca = mapping.get(a["parsed_letter"])
            cb = mapping.get(b["parsed_letter"])
            same = ca is not None and ca == cb
            agreement["n"] += 1
            agreement["same_candidate"] += int(same)
            per_q[qid] = {
                "gt": r["ground_truth_letter"],
                "orig_letter": a["parsed_letter"], "orig_correct": a["correct"],
                "swap_letter": b["parsed_letter"], "swap_correct": b["correct"],
                "orig_candidate": ca, "swap_candidate": cb, "same_candidate": same,
            }
        agreement["ratio"] = (
            agreement["same_candidate"] / agreement["n"] if agreement["n"] else None
        )

        manifest = {
            "schema_version": "architecture_iq.letter_swap_control.v1",
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_run": str(SOURCE_RUN),
            "model": model,
            "notes": (
                "swapped_ab = content of Choice A and Choice B exchanged; "
                "candidate-level agreement measures whether the model follows content, "
                "not letter position."
            ),
            "summary": summary,
            "candidate_agreement": agreement,
            "per_question": per_q,
        }
        (run_dir / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("candidate agreement:", json.dumps(agreement, ensure_ascii=False))
        for qid, row in sorted(per_q.items()):
            print(f"  {qid} GT={row['gt']} orig={row['orig_letter']}({row['orig_candidate']}) "
                  f"swap={row['swap_letter']}({row['swap_candidate']}) "
                  f"same_candidate={row['same_candidate']}")


if __name__ == "__main__":
    main()
