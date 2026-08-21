#!/usr/bin/env python3
"""Training API server for the v1_review viewer.

Serves static files (viewer.html, *.json, curves/, answers/) AND exposes
POST /api/train to run a modified candidate through run_ground_truth.

Usage:
  .venv/bin/python data/v1_review/train_server.py [--port 8502] [--bundle /tmp/v1bundle]
"""
from __future__ import annotations
import argparse, copy, json, os, sys, tempfile, traceback, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
BUNDLE_DEFAULT = Path("/tmp/v1bundle")
DATA_DIR_DEFAULT = WORKTREE / "data" / "v1_review"
SRC = WORKTREE / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MIME = {
    ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".js": "application/javascript", ".css": "text/css", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".md": "text/plain; charset=utf-8",
}


def _set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _coerce(value):
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s in ("True", "true"): return True
    if s in ("False", "false"): return False
    if s.startswith("[") and s.endswith("]"):
        try: return json.loads(s)
        except: pass
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s


def _delta_label(delta):
    if not delta: return "retrain (no change)"
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in delta.items())


def _jsonable(obj):
    if isinstance(obj, dict): return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    try: return float(obj)
    except: return str(obj)


def run_training(payload, bundle):
    from architecture_iq.candidates.generator import write_candidate
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries, get_model_type
    from architecture_iq.util import read_json
    ensure_registries()

    qid = payload["question_id"]
    base_letter = payload["base_choice"]
    delta = payload.get("delta", {})
    n_seeds = int(payload.get("n_seeds", 3))

    qpath = bundle / "benchmarks" / "v1_llm" / "questions" / qid / "question.json"
    qdata = read_json(qpath)
    base = next(c for c in qdata["choices"] if c["letter"] == base_letter)
    cand_dir = bundle / "data" / base["candidate_path"]
    spec = read_json(cand_dir / "candidate_spec.json")
    spec = copy.deepcopy(spec)
    spec.setdefault("execution", {})["device"] = "cpu"  # force CPU on mac
    for k, v in delta.items():
        _set_nested(spec, k, _coerce(v))

    dataset_path = cand_dir.parents[2]
    profile = load_profile("v1_human")
    profile = copy.deepcopy(profile)
    profile.ground_truth["n_seeds"] = n_seeds
    profile.ground_truth["base_seed"] = 0

    tmp = Path(tempfile.mkdtemp(prefix=f"train_{qid}_"))
    try:
        write_candidate(spec, tmp, get_model_type(spec["model"]["type"]))
        summary = run_ground_truth(tmp, profile, dataset_path=dataset_path, fail_threshold_override=float("inf"))
        # aggregate curves
        agg_curve = None
        curves_path = tmp / "results" / "curves.npz"
        if curves_path.exists():
            import numpy as np
            npz = np.load(curves_path, allow_pickle=True)
            per_seed = npz["curves"]
            samples = npz["samples"]
            agg_curve = {
                "letter": f"{base_letter}★",
                "mean": per_seed.mean(axis=0).tolist(),
                "std": per_seed.std(axis=0).tolist(),
                "samples": samples.tolist(),
                "label": _delta_label(delta),
            }
        return {"question_id": qid, "base_choice": base_letter, "delta": delta,
                "summary": _jsonable(summary), "curve": agg_curve, "spec": _jsonable(spec)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    bundle = BUNDLE_DEFAULT
    data_dir = DATA_DIR_DEFAULT

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        rel = parsed.path.lstrip("/")
        if rel == "": rel = "viewer.html"
        # Look in DATA_DIR first (binary_questions.json, curves/, answers/),
        # then HERE (viewer.html, scripts)
        for base in (self.data_dir, HERE):
            path = base / rel
            if path.is_file():
                ct = MIME.get(path.suffix, "application/octet-stream")
                self._send_file(path, ct)
                return
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/train":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"bad payload: {e}"})
            return
        try:
            result = run_training(payload, self.bundle)
            self._send_json(200, result)
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            self._send_json(500, {"error": str(e), "traceback": tb[-1500:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--bundle", default=str(BUNDLE_DEFAULT))
    ap.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
    ap.add_argument("--host", default="127.0.0.1")
    cfg = ap.parse_args()
    Handler.bundle = Path(cfg.bundle)
    Handler.data_dir = Path(cfg.data_dir)
    if not Handler.bundle.is_dir():
        sys.exit(f"bundle dir not found: {cfg.bundle}")
    srv = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    print(f"train server on http://{cfg.host}:{cfg.port}/viewer.html", flush=True)
    print(f"  bundle={Handler.bundle}", flush=True)
    print(f"  data_dir={Handler.data_dir}", flush=True)
    print(f"  POST /api/train {{question_id, base_choice, delta, n_seeds}}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
