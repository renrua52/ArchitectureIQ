"""Score an LLM-proposed config for a ``propose_improvement`` question.

Pipeline (keeps the spec -> code -> run -> GT invariant):

    base config (from question)
      + proposal overrides (LLM JSON)
        -> snapped closed-set spec (schema 2.0)
        -> write_candidate(temp) -> run_ground_truth(temp, profile, problem dir)
        -> compare per-seed with the base candidate's stored results

Closed set (profile v2):
  * batch_size in {16, 32, 64}; total_samples_seen fixed to the base's.
  * optimizer lr / weight_decay / momentum / betas snapped to the grid.
  * model types {mlp, transformer_lm} with profile architecture grids.
  * loss must be compatible with the problem's family.
  * model params <= 1.1 x max params of (base + improved demos + references).

The proposal is interpreted as *overrides* over the base config (missing keys
inherit the base), matching the "LLM proposes a few edits to a base setting"
design.

Usage:
    .venv/bin/python -m backend.eval.score_proposal \\
        --question backend/eval/sets/propose_improvement_v1/questions.jsonl:0 \\
        --proposal artifacts/proposal.json --out artifacts/proposal_score.json
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import Profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.candidates.generator import write_candidate
from architecture_iq.storage import repository as repo
from architecture_iq.util import short_hash

CLOSED_OPTIMIZERS = ("SGD", "Adam", "AdamW", "RMSprop", "Adagrad")
BATCH_SIZES = (16, 32, 64)
LR_GRID = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
WD_GRID = (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
MOMENTUM_GRID = (0.0, 0.9)
BETAS_GRID = (0.9, 0.999)
MAX_PARAM_RATIO = 1.1


def _nearest(value: float, grid: tuple) -> float:
    return min(grid, key=lambda g: abs(float(g) - float(value)))


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def _final_key(metric: str) -> str:
    return f"final_{metric}"


def estimate_params(model: dict) -> float:
    """Rough parameter count for the two supported model families."""
    mtype = model["type"]
    if mtype == "mlp":
        depth = int(model["depth"])
        width = int(model["width"])
        input_dim = int(model.get("input_dim", 1))
        norms = model.get("layer_norm", [False] * depth)
        p = (input_dim + 1) * width  # input linear
        for use_norm in norms[:depth]:
            p += width * width + width  # block linear + bias
            if use_norm:
                p += 2 * width  # layer norm weight + bias
        p += width + 1  # output linear
        return float(p)
    if mtype == "transformer_lm":
        d_model = int(model["d_model"])
        d_ff = int(model["d_ff"])
        num_layers = int(model["num_layers"])
        vocab = int(model["vocab_size"])
        ctx = int(model["context_length"])
        p = vocab * d_model + ctx * d_model  # embeddings
        for _ in range(num_layers):
            p += 3 * d_model * d_model + d_model  # in_proj qkv
            p += d_model * d_model + d_model  # out_proj
            p += 2 * d_model * d_ff + d_ff  # ffn1
            p += d_ff * d_model + d_model  # ffn2
            p += 4 * d_model  # two layer norms
        p += d_model * vocab + vocab  # head
        return float(p)
    raise ValueError(f"unknown model type {mtype!r}")


def _snap_optimizer(opt: dict, notes: list[str]) -> dict:
    out = dict(opt)
    out["lr"] = _nearest(out["lr"], LR_GRID)
    if float(out["lr"]) != float(opt["lr"]):
        notes.append(f"lr {opt['lr']} -> {out['lr']}")
    if "weight_decay" in out:
        out["weight_decay"] = _nearest(out["weight_decay"], WD_GRID)
        if float(out["weight_decay"]) != float(opt["weight_decay"]):
            notes.append(f"weight_decay {opt['weight_decay']} -> {out['weight_decay']}")
    if opt.get("type") == "SGD" and "momentum" in out:
        out["momentum"] = _nearest(out["momentum"], MOMENTUM_GRID)
        if float(out["momentum"]) != float(opt["momentum"]):
            notes.append(f"momentum {opt['momentum']} -> {out['momentum']}")
    if opt.get("type") in ("Adam", "AdamW") and "betas" in out:
        out["betas"] = [_nearest(b, BETAS_GRID) for b in out["betas"]]
        if list(out["betas"]) != list(opt["betas"]):
            notes.append(f"betas {opt['betas']} -> {out['betas']}")
    return out


def normalize_proposal_display(proposal: dict) -> tuple[dict, list[str]]:
    """Map LLM display names to closed-set schema keys.

    LLMs write e.g. ``"type": "causal transformer LM"``, ``"learning_rate"``,
    or ``"loss": "cross-entropy on next-token labels"`` instead of the schema
    values (``transformer_lm`` / ``lr`` / ``{"loss_id": ...}``). This layer
    normalizes those before ``normalize_proposal`` merges them over the base.
    """
    notes: list[str] = []
    p = json.loads(json.dumps(proposal, sort_keys=True))

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    model = p.get("model")
    if isinstance(model, dict) and "type" in model:
        t = str(model["type"])
        n = norm(t)
        if "transformer" in n:
            mapped = "transformer_lm"
        elif "mlp" in n or "perceptron" in n or "linearnet" in n:
            mapped = "mlp"
        else:
            mapped = t
        if mapped != t:
            notes.append(f"model.type {t!r} -> {mapped!r}")
        model["type"] = mapped

    opt = p.get("optimizer")
    if isinstance(opt, dict):
        if "type" in opt:
            t = str(opt["type"])
            n = norm(t)
            alias = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW",
                     "rmsprop": "RMSprop", "adagrad": "Adagrad"}
            mapped = alias.get(n, t)
            if mapped != t:
                notes.append(f"optimizer.type {t!r} -> {mapped!r}")
            opt["type"] = mapped
        lr_key = next((k for k in opt if k.lower() in ("lr", "learning_rate")), None)
        if lr_key is not None and lr_key != "lr":
            opt["lr"] = opt.pop(lr_key)
            notes.append(f"optimizer.{lr_key} -> lr")
        wd_key = next((k for k in opt if k.lower() in ("weight_decay", "weight decay", "wd")), None)
        if wd_key is not None and wd_key != "weight_decay":
            opt["weight_decay"] = opt.pop(wd_key)
            notes.append(f"optimizer.{wd_key} -> weight_decay")

    def map_loss(value: str) -> str:
        n = norm(value)
        if "cross" in n or "softmax" in n or n == "ce":
            lid = "cross_entropy"
        elif "mse" in n or "square" in n or "mean" in n:
            lid = "mse"
        else:
            lid = value
        if "l1" in n:
            lid += "_l1"
        elif "l2" in n:
            lid += "_l2"
        return lid

    loss = p.get("loss")
    if isinstance(loss, str):
        lid = map_loss(loss)
        notes.append(f"loss {loss!r} -> {lid!r}")
        p["loss"] = {"loss_id": lid}
    elif isinstance(loss, dict):
        raw = loss.get("loss_id") or loss.get("type") or loss.get("name")
        if isinstance(raw, str):
            lid = map_loss(raw)
            if lid != raw:
                notes.append(f"loss {raw!r} -> {lid!r}")
            p["loss"] = {"loss_id": lid}

    return p, notes


def normalize_proposal(base_cfg: dict, proposal: dict) -> tuple[dict, list[str], list[str]]:
    """Merge proposal over base, snap to closed set. Returns (spec, notes, errors)."""
    notes: list[str] = []
    errors: list[str] = []
    spec = json.loads(json.dumps(base_cfg, sort_keys=True))

    if "model" in proposal:
        model = dict(proposal["model"])
        mtype = model.get("type", spec["model"]["type"])
        if mtype not in ("mlp", "transformer_lm"):
            errors.append(f"unknown model type {mtype!r}")
            return spec, notes, errors
        model["type"] = mtype
        # keep base's family-required keys unless overridden
        for k, v in spec["model"].items():
            model.setdefault(k, v)
        if mtype == "mlp":
            model["depth"] = int(round(float(model["depth"])))
            model["width"] = int(_nearest(model["width"], (16, 32, 64, 128, 256)))
            if model["depth"] not in range(1, 7):
                errors.append(f"mlp depth {model['depth']} out of [1, 6]")
            if len(model["activations"]) != model["depth"] or len(model["layer_norm"]) != model["depth"]:
                if len(model["activations"]) < model["depth"]:
                    base = model["activations"]
                    model["activations"] = (base * (model["depth"] // max(1, len(base)) + 1))[: model["depth"]]
                else:
                    model["activations"] = model["activations"][: model["depth"]]
                if len(model["layer_norm"]) < model["depth"]:
                    base = model["layer_norm"]
                    model["layer_norm"] = (base * (model["depth"] // max(1, len(base)) + 1))[: model["depth"]]
                else:
                    model["layer_norm"] = model["layer_norm"][: model["depth"]]
                notes.append("mlp activations/layer_norm refit to new depth")
        else:
            model["d_model"] = int(_nearest(model["d_model"], (32, 64, 128)))
            model["num_layers"] = int(_nearest(model["num_layers"], (1, 2, 3)))
            model["num_heads"] = int(_nearest(model["num_heads"], (2, 4)))
            model["d_ff"] = int(_nearest(model["d_ff"], (64, 128, 256)))
            if model["d_model"] % model["num_heads"] != 0:
                errors.append("d_model must be divisible by num_heads")
        spec["model"] = model
        notes.append("model edited")
    else:
        notes.append("model unchanged")

    if "optimizer" in proposal:
        opt = dict(proposal["optimizer"])
        otype = opt.get("type", spec["optimizer"]["type"])
        if otype not in CLOSED_OPTIMIZERS:
            errors.append(f"unknown optimizer type {otype!r}")
            return spec, notes, errors
        opt["type"] = otype
        for k, v in spec["optimizer"].items():
            opt.setdefault(k, v)
        spec["optimizer"] = _snap_optimizer(opt, notes)
        notes.append(f"optimizer={otype}")
    else:
        notes.append(f"optimizer unchanged ({spec['optimizer']['type']})")

    if "loss" in proposal:
        loss = dict(proposal["loss"])
        for k, v in spec["loss"].items():
            loss.setdefault(k, v)  # inherit lambda etc. from the base loss
        if "lambda" not in loss and str(loss.get("loss_id", "")).endswith(("_l1", "_l2")):
            loss["lambda"] = 1.0e-3  # closed-set default from loss_grids.lambda
            notes.append("loss.lambda default 1e-3 (not in base)")
        spec["loss"] = loss
        notes.append(f"loss={loss.get('loss_id')}")
    else:
        notes.append(f"loss unchanged ({spec['loss'].get('loss_id')})")

    if "budget" in proposal:
        budget = dict(proposal["budget"])
        if "batch_size" in budget:
            bs = int(_nearest(budget["batch_size"], BATCH_SIZES))
            if bs != int(budget["batch_size"]):
                notes.append(f"batch_size {budget['batch_size']} -> {bs}")
            budget["batch_size"] = bs
        budget.setdefault("total_samples_seen", spec["budget"]["total_samples_seen"])
        spec["budget"] = budget
        notes.append(f"budget: batch_size={budget.get('batch_size')}, "
                     f"total_samples_seen={budget.get('total_samples_seen')}")
    else:
        notes.append(f"budget unchanged (total_samples_seen="
                     f"{spec['budget']['total_samples_seen']}, "
                     f"batch_size={spec['budget']['batch_size']})")

    # rebuild derived fields
    budget = spec["budget"]
    total = int(budget["total_samples_seen"])
    bs = int(budget["batch_size"])
    if total % bs != 0:
        errors.append(f"total_samples_seen {total} not divisible by batch_size {bs}")
        return spec, notes, errors
    spec["budget"]["training_steps"] = total // bs
    spec["budget"]["batch_size"] = bs
    return spec, notes, errors


def _validate_loss(question: dict, spec: dict) -> list[str]:
    family = spec["family"]
    allowed = {
        "univariate_regression": ("mse", "mse_l2", "mse_l1"),
        "multivariate_regression": ("mse", "mse_l2", "mse_l1"),
        "bigram_lm": ("cross_entropy", "cross_entropy_l2", "cross_entropy_l1"),
    }.get(family, ())
    lid = spec["loss"].get("loss_id")
    if lid not in allowed:
        return [f"loss {lid!r} not compatible with family {family!r} (allowed {allowed})"]
    return []


def _param_cap(question: dict, spec: dict) -> float:
    base = estimate_params(question["base"]["setting"]["model"])
    caps = [base]
    for d in question.get("improved_demos", []):
        caps.append(estimate_params(d["setting"]["model"]))
    for r in question.get("references", []):
        caps.append(estimate_params(r["setting"]["model"]))
    return MAX_PARAM_RATIO * max(caps)


def score_question(question: dict, proposal: dict) -> dict:
    metric = question["metric"]
    base_cfg = question["base"]["setting"]
    base_id = question["base"]["candidate_id"]
    problem_id = question["problem_id"]

    spec, notes, errors = normalize_proposal(base_cfg, proposal)
    errors += _validate_loss(question, spec)
    if errors:
        return {"ok": False, "question_id": question["question_id"], "errors": errors,
                "notes": notes, "snapped_spec": spec}

    # param constraint: model size <= 1.1x max of demos
    try:
        params = estimate_params(spec["model"])
        cap = _param_cap(question, spec)
        constraint_ok = params <= cap
    except (KeyError, ValueError) as e:
        return {"ok": False, "question_id": question["question_id"],
                "errors": [f"param estimation failed: {e}"], "notes": notes}

    spec["candidate_id"] = f"c_{short_hash(spec)}"
    spec["files"] = {"model": "model.py", "train": "train.py",
                     "loss": "loss.py", "optimizer": "optimizer.py"}
    spec["schema_version"] = "2.0"
    spec["problem_id"] = problem_id
    spec.pop("dataset_id", None)

    profile = Profile.load(Path("profiles/v2.yaml"))
    ensure_registries()
    family = get_dataset_family(spec["family"])
    model_family = get_model_type(spec["model"]["type"])
    dataset_path = repo.problem_dir(problem_id)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "candidate"
        write_candidate(spec, out_dir, model_family)
        summary = run_ground_truth(out_dir, profile, dataset_path)

    mean_key = _mean_key(metric)
    final_key = _final_key(metric)
    proposal_loss = summary.get(mean_key)
    if proposal_loss is None:
        return {"ok": False, "question_id": question["question_id"],
                "problem_id": problem_id, "errors": ["proposal excluded: all seeds "
                "failed/diverged in GT run"], "notes": notes,
                "snapped_spec": spec}
    proposal_loss = float(proposal_loss)
    base_sum = repo.read_summary(problem_id, base_id)
    base_loss = float(base_sum[mean_key])

    wins = sum(1 for i in range(summary["n_seeds"])
               if summary["seed_results"][i][final_key] < base_sum["seed_results"][i][final_key])
    n = summary["n_seeds"]
    ratio = max(proposal_loss, base_loss) / max(min(proposal_loss, base_loss), 1e-12)

    return {
        "ok": True,
        "question_id": question["question_id"],
        "problem_id": problem_id,
        "metric": metric,
        "base_candidate": base_id,
        "base_loss": round(base_loss, 6),
        "proposal_loss": round(proposal_loss, 6),
        "ratio_vs_base": round(ratio, 4),
        "win_rate_vs_base": round(wins / n, 3),
        "params": params,
        "params_cap": round(cap, 1),
        "param_constraint_ok": constraint_ok,
        "snapped_spec": spec,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", required=True, help="propose_improvement item JSON, or jsonl:line")
    ap.add_argument("--proposal", required=True, help="proposed config JSON file")
    ap.add_argument("--out", default=None, help="output JSON file")
    args = ap.parse_args()

    qpath = args.question
    if ":" in qpath and qpath.rsplit(":", 1)[1].isdigit():
        qfile, line = qpath.rsplit(":", 1)
        question = json.loads(Path(qfile).read_text(encoding="utf-8").splitlines()[int(line)])
    else:
        question = json.loads(Path(qpath).read_text(encoding="utf-8"))
    proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))

    result = score_question(question, proposal)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
