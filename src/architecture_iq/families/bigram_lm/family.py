from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from architecture_iq.families.base import DatasetFamily
from architecture_iq.profile import Profile
from architecture_iq.runtime.loader import load_synthesize_module
from architecture_iq.util import short_hash, write_json


SYNTHESIZE_TEMPLATE = '''"""Bigram LM dataset — one shared transition matrix for train and test."""
from __future__ import annotations

import numpy as np
import torch


def target(seq: torch.Tensor) -> torch.Tensor:
    """Next-token labels: input sequence shifted by one (seq shape [N, L+1] -> [N, L])."""
    return seq[:, 1:].to(torch.int64)


def build_transition(
    *,
    table_seed: int = {table_seed},
    vocab_size: int = {vocab_size},
    alpha: float = {alpha},
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(table_seed)
    logits = rng.standard_normal((vocab_size, vocab_size)) * alpha
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = (probs / probs.sum(axis=1, keepdims=True)).astype(np.float64)

    pi = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    for _ in range(256):
        pi = pi @ probs
        pi /= pi.sum()
    return probs, pi


def synthesize(
    *,
    train_size: int = {train_size},
    test_size: int = {test_size},
    sequence_seed: int = {sequence_seed},
    table_seed: int = {table_seed},
    vocab_size: int = {vocab_size},
    context_length: int = {context_length},
    alpha: float = {alpha},
    layout: str = {layout!r},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_sequences(n: int, length: int, probs: np.ndarray, pi: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        v = probs.shape[0]
        seq = np.empty((n, length), dtype=np.int64)
        for i in range(n):
            s = [int(rng.choice(v, p=pi))]
            while len(s) < length:
                s.append(int(rng.choice(v, p=probs[s[-1]])))
            seq[i] = np.asarray(s, dtype=np.int64)
        return seq

    probs, pi = build_transition(
        table_seed=table_seed,
        vocab_size=vocab_size,
        alpha=alpha,
    )
    if layout == "lm":
        seq_train = _sample_sequences(train_size, context_length + 1, probs, pi, sequence_seed)
        seq_test = _sample_sequences(test_size, context_length + 1, probs, pi, sequence_seed + 1)
    else:
        raise ValueError("Only layout='lm' is supported in generated synthesize.py")

    train_seq = torch.from_numpy(seq_train)
    test_seq = torch.from_numpy(seq_test)
    x_train = train_seq[:, :-1].to(torch.int64)
    y_train = target(train_seq)
    x_test = test_seq[:, :-1].to(torch.int64)
    y_test = target(test_seq)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
'''


class BigramLmFamily(DatasetFamily):
    name = "bigram_lm"

    @staticmethod
    def _rng_streams(instance_seed: int) -> tuple[int, int]:
        return instance_seed, instance_seed + 10_000

    def create_instance(self, profile: Profile, seed: int) -> dict[str, Any]:
        sequence_seed, table_seed = self._rng_streams(seed)
        cfg = profile.family_config(self.name)
        params = {
            "instance_seed": seed,
            "vocab_size": int(cfg["vocab_size"]),
            "context_length": int(cfg["context_length"]),
            "train_size": int(cfg["train_size"]),
            "test_size": int(cfg["test_size"]),
            "alpha": float(cfg.get("alpha", 1.0)),
            "layout": str(cfg.get("layout", "lm")),
            "sequence_seed": sequence_seed,
            "table_seed": table_seed,
        }
        significance = {
            "gap_min": float(profile.significance["gap_min"]),
            "fail_threshold": float(cfg.get("fail_threshold", profile.ground_truth["fail_threshold"])),
        }
        return {
            "schema_version": profile.schema_version,
            "family": self.name,
            "params": params,
            "selection_metric": "test_ce",
            "significance": significance,
            "files": {
                "synthesize": "synthesize.py",
                "train": "train.pt",
                "test": "test.pt",
                "transition": "transition.npz",
            },
        }

    def build_spec_with_id(self, partial: dict[str, Any]) -> dict[str, Any]:
        dataset_id = f"bg_{short_hash(partial['params'])}"
        spec = {k: v for k, v in partial.items() if not k.startswith("_")}
        spec["dataset_id"] = dataset_id
        return spec

    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = spec["params"]
        synth_code = SYNTHESIZE_TEMPLATE.format(
            train_size=params["train_size"],
            test_size=params["test_size"],
            sequence_seed=params["sequence_seed"],
            table_seed=params["table_seed"],
            vocab_size=params["vocab_size"],
            context_length=params["context_length"],
            alpha=params["alpha"],
            layout=params["layout"],
        )
        (out_dir / "synthesize.py").write_text(synth_code, encoding="utf-8")

        module = load_synthesize_module(out_dir / "synthesize.py")
        tx, ty, vx, vy = module.synthesize()
        probs, pi = module.build_transition()

        torch.save({"x": tx, "y": ty}, out_dir / "train.pt")
        torch.save({"x": vx, "y": vy}, out_dir / "test.pt")
        np.savez(
            out_dir / "transition.npz",
            probs=probs,
            pi=pi,
        )
        write_json(out_dir / "dataset_spec.json", spec)

    def load_tensors(self, dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train = torch.load(dataset_path / "train.pt", weights_only=True)
        test = torch.load(dataset_path / "test.pt", weights_only=True)
        return train["x"], train["y"], test["x"], test["y"]

    def selection_metric_name(self) -> str:
        return "test_ce"

    def default_significance(self) -> dict[str, Any]:
        return {}

    def compatible_model_types(self) -> list[str]:
        return ["transformer_lm"]
