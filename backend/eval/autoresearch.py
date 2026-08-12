"""AutoResearch propose-loop evaluation (L2 in docs/plan-autoresearch-eval.md).

Loop per question (one work tree):
    round 1..K:
        prompt = tree state (base + lit losses + unlit summaries + budget rule)
        model  -> proposed config JSON (closed set; may equal an unlit node)
        normalize/validate -> if a stored result exists: light it (reveal);
                              else run GT (write_candidate + run_ground_truth),
                              store result + config into backend/data, light it.
        observe loss -> next round.
Scoring:
    improve_base  = 1 - best_loss / base_loss
    oracle_gap    = (best_loss - oracle) / base_loss        (oracle = best tree loss)
    regret@k      = (best_loss_k - oracle) / base_loss
    win_rate_vs_base, params_constraint_ok, new_gt_runs

Usage:
    .venv/bin/python -m backend.eval.autoresearch --tree mvar_866b4e/tree_xxxx \
        --model gpt-5.6-luna --rounds 5
    .venv/bin/python -m backend.eval.autoresearch --build-trees --trees-per-problem 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
from pathlib import Path

import httpx

from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import Profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.candidates.generator import write_candidate
from architecture_iq.storage import repository as repo
from architecture_iq.util import short_hash

from backend.eval import score_proposal, worktree
from backend.eval.batch_eval import call_llm, relay_config, resolve_api_key
from backend.eval.batch_propose import extract_json

OUT_ROOT = Path("artifacts/autoresearch_runs")
N_LIT_FEWSHOT = 3          # lit children revealed as few-shot in round 1
N_UNLIT_SHOWN = 6          # unlit nodes listed in the prompt (capped)
MAX_PARAM_RATIO = 1.1


def _ensure_v1(base_url: str) -> str:
    """Normalize a relay host or full URL to a /v1 chat-completions base."""
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def _final_key(metric: str) -> str:
    return f"final_{metric}"


def _model_line(tree: dict, n: dict, with_loss: bool) -> str:
    cfg = repo.read_candidate_config(tree["problem_id"], n["candidate_id"])
    label = "BASE" if n.get("role") == "base" else f"node {n['candidate_id']}"
    line = f"- {label}: {worktree.setting_line(tree, n)}"
    if with_loss and n.get("loss") is not None:
        line += f"  => loss = {n['loss']:.4g}"
    return line


def build_loop_prompt(tree: dict, lit_ids: set[str], round_no: int, total: int,
                       discovered: list[dict] | None = None) -> str:
    metric = tree["metric"]
    view = worktree.tree_view(tree, lit_ids)
    base = next(n for n in tree["nodes"] if n["candidate_id"] == tree["base"])

    lines = [
        "You are an AutoResearch agent. You are improving the training config of a toy",
        "machine-learning experiment, one config edit at a time. A 'config' is a JSON",
        "with keys model / optimizer / loss / budget. You will make a proposal, then",
        "receive the measured loss, and iterate.",
        "",
        f"[Dataset]  problem: {tree['problem_id']}  metric: {metric} (lower is better)",
        "",
        "[Work tree state]",
        _model_line(tree, base, True),
    ]
    # lit children with losses (few-shot calibration inside this decision tree)
    for n in sorted(view["lit"], key=lambda x: x["loss"]):
        if n["candidate_id"] == tree["base"]:
            continue
        lines.append(_model_line(tree, n, True))
    # experiments the agent ran itself in earlier rounds
    for n in (discovered or []):
        lines.append(f"- new node {n['candidate_id']} (your experiment): "
                     f"{n.get('summary', '')}  => loss = {n['loss']:.4g}")
    # unlit nodes (configs available but not yet tested)
    if view["unlit"]:
        lines.append("")
        lines.append("[Unlit configs already proposed in this tree (NOT yet tested; "
                     "you may light one by proposing its exact config, or propose something new)]")
        for n in view["unlit"][:N_UNLIT_SHOWN]:
            lines.append(_model_line(tree, n, False))
    lines.append("")
    lines.append(f"[Budget rule]  model params <= {MAX_PARAM_RATIO}x the base's params "
                 f"({tree['budget_rule']['total_samples_seen']} total samples seen, "
                 "fixed budget). Do not change total_samples_seen.")
    lines.append("")
    lines.append(f"[Round {round_no}/{total}]")
    lines.append("Propose ONE config as a JSON object (keys: model / optimizer / loss / "
                 "budget.batch_size). Output only the JSON.")
    lines.append("You have TWO legal actions:")
    lines.append("  (a) LIGHT an unlit config: propose its exact JSON (copy it); its loss will be revealed.")
    lines.append("  (b) PROPOSE a new config: make a meaningful edit to the base or a lit config "
                 "(1-2 fields); a new experiment will run and its loss will be revealed.")
    lines.append("Proposing a config that is already lit (including the base) wastes a round "
                 "-- avoid it unless you deliberately conclude no change is worthwhile.")
    return "\n".join(lines)


def build_initial_state(tree: dict, rng: random.Random) -> tuple[set[str], list[str]]:
    """lit = {base} + few-shot lit children; returns (lit_ids, few_shot_ids)."""
    lit = {tree["base"]}
    few = worktree.pick_few_shot(tree, {n["candidate_id"] for n in tree["nodes"]}
                                 - {tree["base"]}, rng, k=N_LIT_FEWSHOT)
    # Few-shot = lit from round 1; they stay visible for the whole loop.
    lit.update(few)
    return lit, few


def _config_equal(a: dict, b: dict) -> bool:
    """Recursive equality tolerant of int/float differences (0 vs 0.0).

    ``candidate_id`` is a derived field, not part of the config identity.
    """
    DERIVED = {"candidate_id", "problem_id", "schema_version", "files", "dataset_id"}
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {k for k in a if k not in DERIVED}
        kb = {k for k in b if k not in DERIVED}
        if ka != kb:
            return False
        return all(_config_equal(a[k], b[k]) for k in ka)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def find_stored_candidate(tree: dict, spec: dict) -> str | None:
    """Return the candidate_id whose stored config equals ``spec``, or None."""
    for cid in repo.list_candidate_ids(tree["problem_id"]):
        try:
            cfg = repo.read_candidate_config(tree["problem_id"], cid)
        except FileNotFoundError:
            continue
        if _config_equal(cfg, spec):
            return cid
    return None


def _stored_loss(tree: dict, candidate_id: str) -> float | None:
    try:
        s = repo.read_summary(tree["problem_id"], candidate_id)
    except FileNotFoundError:
        return None
    mean_key = _mean_key(tree["metric"])
    if s.get("excluded") or mean_key not in s:
        return None
    return float(s[mean_key])


def run_new_gt(tree: dict, spec: dict) -> tuple[dict, dict]:
    """Execute a genuinely new proposed config and persist result + config.

    Strips derived/identity keys inherited from the base config so the new
    candidate never collides with the base's id (bugfix: 2026-08-02).
    """
    spec = dict(spec)
    for k in ("candidate_id", "problem_id", "schema_version", "files", "dataset_id"):
        spec.pop(k, None)
    spec["candidate_id"] = f"c_{short_hash(spec)}"
    spec["schema_version"] = "2.0"
    spec["problem_id"] = tree["problem_id"]
    spec.setdefault("files", {"model": "model.py", "train": "train.py",
                              "loss": "loss.py", "optimizer": "optimizer.py"})

    profile = Profile.load(Path("profiles/v2.yaml"))
    ensure_registries()
    model_family = get_model_type(spec["model"]["type"])
    dataset_path = repo.problem_dir(tree["problem_id"])

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "candidate"
        write_candidate(spec, out_dir, model_family)
        summary = run_ground_truth(out_dir, profile, dataset_path)

    repo.write_candidate_config(tree["problem_id"], spec)
    res_dir = repo.results_dir(tree["problem_id"], spec["candidate_id"])
    res_dir.mkdir(parents=True, exist_ok=True)
    from architecture_iq.util import write_json
    write_json(res_dir / "summary.json", summary)
    return spec, summary


def _win_rate_vs_base(tree: dict, candidate_id: str, base_sum: dict) -> float:
    try:
        s = repo.read_summary(tree["problem_id"], candidate_id)
    except FileNotFoundError:
        return 0.0
    final_key = _final_key(tree["metric"])
    n = min(len(s["seed_results"]), len(base_sum["seed_results"]))
    if n == 0:
        return 0.0
    wins = sum(1 for i in range(n)
               if s["seed_results"][i][final_key] < base_sum["seed_results"][i][final_key])
    return wins / n


async def _run_one_loop_with_client(tree: dict, model: str, client: httpx.AsyncClient,
                                    sem: asyncio.Semaphore, *, rounds: int = 5,
                                    rng: random.Random) -> dict:
    """Run one tree's propose-loop sharing the batch client + semaphore
    (keeps the relay load bounded: one shared pool instead of one pool per tree)."""
    metric = tree["metric"]
    mean_key = _mean_key(metric)
    base_sum = repo.read_summary(tree["problem_id"], tree["base"])
    base_loss = float(base_sum[mean_key])
    oracle = float(tree["oracle"])

    lit, few = build_initial_state(tree, rng)
    discovered: list[dict] = []
    history: list[dict] = []
    best_id, best_loss = tree["base"], base_loss
    new_gt_runs = 0

    for r in range(1, rounds + 1):
            prompt = build_loop_prompt(tree, lit, r, rounds, discovered)
            content, reasoning = await call_llm(client, sem, prompt, model)
            raw = content or ""
            rec = {"round": r, "prompt": prompt, "raw": raw, "reasoning": reasoning,
                   "lit_before": sorted(lit), "proposal": None, "ok": False,
                   "loss": None, "notes": [], "errors": [], "wasted": False}
            prop = extract_json(raw) if raw else None
            if prop is None:
                rec["errors"].append("json parse failed")
                history.append(rec)
                continue
            rec["proposal"] = prop
            try:
                prop2, disp_notes = score_proposal.normalize_proposal_display(prop)
                base_cfg = repo.read_candidate_config(tree["problem_id"], tree["base"])
                spec, notes, errors = score_proposal.normalize_proposal(base_cfg, prop2)
                rec["notes"] = disp_notes + notes
                # validate closed set
                errors += score_proposal._validate_loss(
                    {"base": {"setting": base_cfg}}, spec)
                if spec["budget"]["total_samples_seen"] != tree["budget_rule"]["total_samples_seen"]:
                    errors.append("total_samples_seen must stay fixed to the base's")
                params = score_proposal.estimate_params(spec["model"])
                cap = MAX_PARAM_RATIO * tree["nodes"][0]["params"] if tree["nodes"][0]["params"] else None
                constraint_ok = cap is None or params <= cap
                if not constraint_ok:
                    errors.append(f"params {params:.0f} > 1.1x base cap {cap:.0f}")
                rec["params"], rec["params_cap"], rec["constraint_ok"] = params, cap, constraint_ok
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(f"normalize failed: {e}")
                history.append(rec)
                continue
            if errors:
                rec["errors"] += errors
                history.append(rec)
                continue

            match = find_stored_candidate(tree, spec)
            if match is not None:
                cid = match
                loss = _stored_loss(tree, cid)
                rec["lit_existing"] = True
            else:
                # GT is CPU-bound; run it off the event loop so the batch's
                # concurrent LLM calls keep flowing (bugfix 2026-08-02).
                spec2, summary = await asyncio.to_thread(run_new_gt, tree, spec)
                cid = spec2["candidate_id"]
                loss = _stored_loss(tree, cid)
                new_gt_runs += 1
                rec["lit_existing"] = False
                rec["notes"] += [f"new experiment run: {cid}"]
            rec["ok"] = loss is not None
            rec["loss"] = loss
            rec["candidate_id"] = cid
            rec["wasted"] = cid in rec["lit_before"]
            if loss is not None:
                lit.add(cid)
                if not rec["lit_existing"]:
                    discovered.append({"candidate_id": cid, "loss": loss,
                                       "summary": rec["notes"][:1] and " ".join(rec["notes"][:3])})
                if loss < best_loss:
                    best_id, best_loss = cid, loss
            rec["best_loss"] = best_loss
            rec["improve_base"] = round(1 - best_loss / base_loss, 4) if base_loss else None
            rec["regret"] = round((best_loss - oracle) / base_loss, 4) if base_loss else None
            rec["win_rate_vs_base"] = round(_win_rate_vs_base(tree, best_id, base_sum), 3)
            history.append(rec)

    return {
        "schema_version": "0.1",
        "tree_id": tree["tree_id"],
        "problem_id": tree["problem_id"],
        "metric": metric,
        "model": model,
        "rounds": rounds,
        "base": tree["base"],
        "base_loss": round(base_loss, 6),
        "oracle": round(oracle, 6),
        "beat_tree_oracle": bool(best_loss < oracle - 1e-12),
        "best_candidate": best_id,
        "best_loss": round(best_loss, 6),
        "improve_base": round(1 - best_loss / base_loss, 4) if base_loss else None,
        "oracle_gap_rel": round((best_loss - oracle) / base_loss, 4) if base_loss else None,
        "new_gt_runs": new_gt_runs,
        "few_shot": few,
        "history": history,
    }


async def run_one_loop(tree: dict, model: str, *,
                       rounds: int = 5, concurrency: int = 1,
                       base_url: str | None = None, api_key: str | None = None,
                       rng: random.Random | None = None,
                       client: httpx.AsyncClient | None = None,
                       sem: asyncio.Semaphore | None = None) -> dict:
    """Run one tree's propose-loop. Pass a shared ``client``/``sem`` for batch
    runs; otherwise an isolated client + semaphore is created."""
    rng = rng or random.Random(0)
    key = api_key or resolve_api_key()
    base_url = _ensure_v1(base_url or relay_config()["eval"]["base_url"])
    headers = {"Authorization": f"Bearer {key}"}
    if client is not None and sem is not None:
        return await _run_one_loop_with_client(tree, model, client, sem,
                                               rounds=rounds, rng=rng)
    sem = sem or asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as own:
        return await _run_one_loop_with_client(tree, model, own, sem,
                                               rounds=rounds, rng=rng)


def persist_run(run: dict) -> Path:
    run_id = f"{run['problem_id']}_{run['tree_id']}_{short_hash(json.dumps(run['history'], sort_keys=True))[:8]}"
    out = OUT_ROOT / run["model"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in run.items() if k != "history"}
    (out / "summary.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "history.jsonl").open("w", encoding="utf-8") as f:
        for rec in run["history"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


async def _amain(args: argparse.Namespace) -> int:
    if args.build_trees:
        return _build_trees(args)
    if args.trees:
        refs = args.trees.split(",")
    elif args.limit:
        refs = sorted(str(p.parent.name) + "/" + p.stem
                      for p in worktree.TREES_ROOT.glob("*/*.json"))[: args.limit]
    else:
        refs = [args.tree]
    rng = random.Random(args.seed)
    key = resolve_api_key()
    base_url = _ensure_v1(args.base_url or relay_config()["eval"]["base_url"])
    headers = {"Authorization": f"Bearer {key}"}
    sem = asyncio.Semaphore(args.concurrency)

    async def run_and_persist(client: httpx.AsyncClient, ref: str) -> dict:
        pid, tid = ref.split("/", 1)
        run = await _run_one_loop_with_client(worktree.load_tree(pid, tid), args.model,
                                              client, sem, rounds=args.rounds, rng=rng)
        out = persist_run(run)
        print(f"[done] {tid}: improve={run['improve_base']} new_gt={run['new_gt_runs']} -> {out}",
              flush=True)
        return run

    runs: list[dict] = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers,
                                 limits=httpx.Limits(max_connections=args.concurrency * 2)) as client:
        tasks = [asyncio.create_task(run_and_persist(client, r)) for r in refs]
        for coro in asyncio.as_completed(tasks):
            runs.append(await coro)
    runs.sort(key=lambda r: r["tree_id"])
    summary = {r["tree_id"]: {k: r[k] for k in (
        "problem_id", "base_loss", "best_loss", "improve_base", "oracle_gap_rel",
        "new_gt_runs", "best_candidate")} for r in runs}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _build_trees(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    pids = args.problems.split(",") if args.problems else repo.list_problems()
    total = 0
    for pid in pids:
        made = 0
        for _ in range(args.trees_per_problem):
            t = worktree.build_tree(pid, rng)
            if t is None:
                continue
            worktree.save_tree(t)
            made += 1
        total += made
        print(f"{pid}: {made} trees")
    print(f"total trees: {total}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", help="problem_id/tree_id (single)")
    ap.add_argument("--trees", help="comma-separated problem_id/tree_id refs")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N trees across all problems (sorted)")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--build-trees", action="store_true")
    ap.add_argument("--problems", default=None)
    ap.add_argument("--trees-per-problem", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()
    if not args.build_trees and not (args.tree or args.trees or args.limit):
        ap.error("need --tree/--trees/--limit or --build-trees")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
