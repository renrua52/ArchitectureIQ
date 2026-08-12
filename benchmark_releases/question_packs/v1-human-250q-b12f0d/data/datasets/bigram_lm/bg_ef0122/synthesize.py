"""Bigram LM dataset — one shared transition matrix for train and test."""
from __future__ import annotations

import numpy as np
import torch


def target(seq: torch.Tensor) -> torch.Tensor:
    """Next-token labels: input sequence shifted by one (seq shape [N, L+1] -> [N, L])."""
    return seq[:, 1:].to(torch.int64)


def build_transition(
    *,
    table_seed: int = 11018,
    vocab_size: int = 32,
    alpha: float = 1.0,
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
    train_size: int = 800,
    test_size: int = 200,
    sequence_seed: int = 1018,
    table_seed: int = 11018,
    vocab_size: int = 32,
    context_length: int = 16,
    alpha: float = 1.0,
    layout: str = 'lm',
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
