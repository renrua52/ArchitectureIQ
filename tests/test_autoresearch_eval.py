"""Tests for the AutoResearch work-tree eval (docs/plan-autoresearch-eval.md).

Covers: tree building invariants, tree view lit/unlit split, loop prompt
rendering, and the propose-loop scoring on a real backend problem with a
patched (fake) LLM so no network calls are made.
"""
from __future__ import annotations

import asyncio
import json
import random

import pytest

from backend.eval import autoresearch, worktree

PID = "mvar_866b4e"


def _tree() -> dict:
    # need enough children so the loop has unlit nodes to light
    for seed in range(40):
        tree = worktree.build_tree(PID, random.Random(seed), n_children=8)
        if tree is not None and len(tree["nodes"]) >= 6:
            return tree
    pytest.fail("no rich tree found for fixture problem")


def test_build_tree_invariants():
    tree = _tree()
    nodes = tree["nodes"]
    assert nodes[0]["role"] == "base"
    assert len(nodes) >= 4
    assert tree["base"] == nodes[0]["candidate_id"]
    # base is not the oracle: children exist and at least one is better
    base_loss = tree["base_loss"]
    assert tree["oracle"] <= base_loss
    assert any(n["loss"] < base_loss for n in nodes[1:])
    # every child differs from base by 1-2 salient edits
    base_cfg = __import__("architecture_iq.storage.repository", fromlist=["repo"]).read_candidate_config(
        PID, tree["base"])
    from backend.eval.questions import salient_distance
    for n in nodes[1:]:
        cfg = __import__("architecture_iq.storage.repository", fromlist=["repo"]).read_candidate_config(
            PID, n["candidate_id"])
        assert 1 <= salient_distance(base_cfg, cfg) <= 3
        assert n["loss"] > 0


def test_tree_view_lit_unlit():
    tree = _tree()
    lit = {tree["base"], tree["nodes"][1]["candidate_id"]}
    view = worktree.tree_view(tree, lit)
    assert len(view["lit"]) == 2
    assert all("loss" in n for n in view["lit"])
    assert all("loss" not in n for n in view["unlit"])
    assert len(view["lit"]) + len(view["unlit"]) == len(tree["nodes"])


def test_loop_prompt_contains_tree_state():
    tree = _tree()
    lit, few = autoresearch.build_initial_state(tree, random.Random(0))
    prompt = autoresearch.build_loop_prompt(tree, lit, 1, 5)
    assert tree["problem_id"] in prompt
    assert tree["metric"] in prompt
    assert "BASE" in prompt
    assert "Unlit configs" in prompt
    assert "Round 1/5" in prompt


def _make_run_tree() -> dict:
    # deterministic small tree from real stored GT; iterate over seeds so the
    # test stays meaningful as the stored candidate pool grows
    from architecture_iq.storage import repository as repo
    for seed in range(200):
        tree = worktree.build_tree(PID, random.Random(seed), n_children=8)
        if tree is None or len(tree["nodes"]) < 6:
            continue
        lit, _ = autoresearch.build_initial_state(tree, random.Random(0))
        base_cfg = repo.read_candidate_config(PID, tree["base"])
        unlit = [n for n in tree["nodes"] if n["candidate_id"] not in lit]
        # require a budget-compliant unlit child that actually beats base
        good = [n for n in unlit
                if worktree.budget_ok(base_cfg, repo.read_candidate_config(PID, n["candidate_id"]))
                and n["loss"] < tree["base_loss"]]
        if good:
            return tree, lit
    pytest.fail("no tree with a budget-compliant unlit child")


def test_loop_with_invalid_proposals_does_not_crash():
    tree, lit = _make_run_tree()

    async def fake_call_llm(client, sem, prompt, model, **_):
        return "not json at all", "thinking..."

    orig = autoresearch.call_llm
    autoresearch.call_llm = fake_call_llm
    try:
        run = asyncio.run(autoresearch.run_one_loop(
            tree, "fake-model", rounds=2, concurrency=1,
            base_url="http://localhost:9/v1", api_key="x", rng=random.Random(0)))
    finally:
        autoresearch.call_llm = orig
    assert run["rounds"] == 2
    assert len(run["history"]) == 2
    assert all(not r["ok"] for r in run["history"])
    assert abs(run["best_loss"] - tree["base_loss"]) < 1e-5  # no improvement without valid moves
    assert run["new_gt_runs"] == 0


def test_loop_lights_unlit_child_and_improves():
    tree, lit = _make_run_tree()
    # target: best unlit child config that respects the 1.1x budget rule
    from architecture_iq.storage import repository as repo
    base_cfg = repo.read_candidate_config(PID, tree["base"])
    unlit = [n for n in tree["nodes"] if n["candidate_id"] not in lit]
    target = min((n for n in unlit
                  if worktree.budget_ok(base_cfg, repo.read_candidate_config(PID, n["candidate_id"]))),
                 key=lambda n: n["loss"])
    target_cfg = repo.read_candidate_config(PID, target["candidate_id"])
    target_json = json.dumps(target_cfg)

    async def fake_call_llm(client, sem, prompt, model, **_):
        return target_json, "propose best unlit config"

    orig = autoresearch.call_llm
    autoresearch.call_llm = fake_call_llm
    try:
        run = asyncio.run(autoresearch.run_one_loop(
            tree, "fake-model", rounds=3, concurrency=1,
            base_url="http://localhost:9/v1", api_key="x", rng=random.Random(0)))
    finally:
        autoresearch.call_llm = orig
    assert run["best_loss"] <= target["loss"] + 1e-5
    assert run["improve_base"] is not None and run["improve_base"] > 0
    assert run["oracle_gap_rel"] >= 0
    # all three rounds attempted; final best recorded
    assert all(r["raw"] for r in run["history"])
    assert abs(run["history"][-1]["best_loss"] - run["best_loss"]) < 1e-5


def test_loop_new_proposal_dedup_and_no_base_overwrite():
    """Re-proposing the same NEW config runs GT once; base config is never
    overwritten by a new candidate's id."""
    from architecture_iq.storage import repository as repo
    tree, lit = _make_run_tree()
    base_cfg = repo.read_candidate_config(PID, tree["base"])
    base_sum_before = repo.read_summary(PID, tree["base"])
    # a genuinely new, budget-compliant config: swap SGD -> AdamW on the base
    new_cfg = json.loads(json.dumps(base_cfg))
    new_cfg["optimizer"]["type"] = "AdamW"
    new_json = json.dumps(new_cfg)

    async def fake_call_llm(client, sem, prompt, model, **_):
        return new_json, "swap optimizer"

    orig = autoresearch.call_llm
    autoresearch.call_llm = fake_call_llm
    try:
        run = asyncio.run(autoresearch.run_one_loop(
            tree, "fake-model", rounds=3, concurrency=1,
            base_url="http://localhost:9/v1", api_key="x", rng=random.Random(0)))
    finally:
        autoresearch.call_llm = orig
    assert run["new_gt_runs"] == 1, f"expected exactly 1 new GT, got {run['new_gt_runs']}"
    assert all(r["lit_existing"] or r["candidate_id"] == run["best_candidate"] or not r["ok"]
               for r in run["history"])
    # base config + summary untouched
    assert repo.read_candidate_config(PID, tree["base"]) == base_cfg
    assert repo.read_summary(PID, tree["base"]) == base_sum_before
    # the new candidate was stored under its own id, not the base's
    cids = [r["candidate_id"] for r in run["history"] if r.get("candidate_id")]
    assert all(c != tree["base"] for c in cids)
    # clean up the candidate this test created (keeps the test idempotent)
    for c in cids:
        cfg_path = repo.candidate_config_path(PID, c)
        res_dir = repo.results_dir(PID, c)
        if cfg_path.exists() and c not in {n["candidate_id"] for n in tree["nodes"]}:
            cfg_path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(res_dir, ignore_errors=True)
