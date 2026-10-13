"""Convert AIQ question rows into rlforge trainer rows that match the SFT prompt layout.

SFT (sft_rc/train_sft_rc.py) trained on a single USER message rendered with
apply_chat_template(add_generation_prompt=True, enable_thinking=True). The kb pool /
release rows carry an extra one-line system message for ranking and dataflip; it is
dropped by default so the RL prompt is token-identical in layout to SFT (the user
content already contains the full answer-format section). --keep-system disables this.

Output row: {prompt:[{role:user,content}], answer, source, question_id, task,
num_choices, family}. Flat scalars only (they become reward kwargs).

arch170 (kb_science_pool_v1 / rl_pool_b_v0) keeps the v1.5 original contract
(<explanation> + <answer>, user only). --arch170-canonical OLD_POOL replaces its messages by
the canonicalized v15-main version of the same question_id from the old 3,830 pool
(train_v3.jsonl: system + user, single <answer> tag), so every select source shares one
output contract. Answers are asserted equal.

Usage:
  python prep_rows.py convert IN.jsonl OUT.jsonl [--keep-system] [--arch170-canonical train_v3.jsonl]
  python prep_rows.py sample  POOL.jsonl OUTDIR --sftrc-ids sftrc_ids.json [--arch170-canonical ...]
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 20261003


CANON = {}


def load_canon(path):
    if path:
        for line in open(path):
            o = json.loads(line)
            if o.get("source") == "v15-main":
                CANON[o["question_id"]] = o


def to_row(r, keep_system=False):
    msgs = r.get("messages") or r.get("prompt") or r.get("prompt_messages")
    if CANON and r.get("source") == "arch170":
        o = CANON[r["question_id"]]
        assert str(o["answer"]).strip() == str(r["answer"]).strip(), r["question_id"]
        msgs = o["prompt"]
    if isinstance(msgs, str):
        msgs = [{"role": "user", "content": msgs}]
    if not keep_system:
        msgs = [m for m in msgs if m["role"] != "system"]
    assert msgs and msgs[-1]["role"] == "user", r.get("question_id")
    ans = str(r["answer"]).strip()
    task = r.get("task") or ("ranking" if "<" in ans else "select")
    return {
        "prompt": [{"role": m["role"], "content": m["content"]} for m in msgs],
        "answer": ans,
        "source": r.get("source", "?"),
        "question_id": r["question_id"],
        "task": "ranking" if task == "ranking" else "select",
        "num_choices": int(r.get("num_choices") or (5 if "<" in ans else 3)),
        "family": r.get("family", "?"),
    }


def write(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows), hashlib.md5(path.read_bytes()).hexdigest(),
          dict(Counter(r["source"] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("inp"); c.add_argument("out"); c.add_argument("--keep-system", action="store_true")
    c.add_argument("--arch170-canonical", default="")
    s = sub.add_parser("sample")
    s.add_argument("pool"); s.add_argument("outdir"); s.add_argument("--sftrc-ids", required=True)
    s.add_argument("--arch170-canonical", default="")
    a = ap.parse_args()
    load_canon(a.arch170_canonical)
    if a.cmd == "convert":
        write([to_row(json.loads(l), a.keep_system) for l in open(a.inp) if l.strip()], a.out)
        return
    pool = [json.loads(l) for l in open(a.pool)]
    ids = json.load(open(a.sftrc_ids))
    seen = set(ids["train"]) | set(ids["heldout"])
    rng = random.Random(SEED)
    by = {}
    for r in pool:
        by.setdefault(r["source"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["question_id"])
    arch_unseen = [r for r in by["arch170"] if r["question_id"] not in seen]
    probe = (rng.sample(by["ranking_v3"], 60) + rng.sample(by["ranking_v2"], 60) + rng.sample(arch_unseen, 40))
    used = {r["question_id"] for r in probe}
    left = lambda src, pool_=None: [r for r in (pool_ or by[src]) if r["question_id"] not in used]
    smoke = (rng.sample(left("ranking_v3"), 24) + rng.sample(left("ranking_v2"), 16)
             + rng.sample(left("arch170", arch_unseen), 12) + rng.sample(left("dataflip500"), 12))
    rng.shuffle(smoke)
    write([to_row(r) for r in probe], Path(a.outdir) / "probe160.jsonl")
    write([to_row(r) for r in smoke], Path(a.outdir) / "smoke64.jsonl")
    print("arch170 never in SFT train/heldout:", len(arch_unseen))


if __name__ == "__main__":
    main()
