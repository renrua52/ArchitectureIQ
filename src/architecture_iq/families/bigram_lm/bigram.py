"""Bigram transition utilities for next-token prediction datasets."""

from __future__ import annotations

import numpy as np


def make_bigram_transition(
    vocab_size: int,
    *,
    rng: np.random.Generator,
    alpha: float = 1.0,
) -> np.ndarray:
    v = int(vocab_size)
    logits = rng.standard_normal((v, v))
    logits = alpha * logits
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def stationary_distribution(probs: np.ndarray, iters: int = 256) -> np.ndarray:
    v = probs.shape[0]
    pi = np.full(v, 1.0 / v, dtype=np.float64)
    for _ in range(iters):
        pi = pi @ probs
        pi /= pi.sum()
    return pi


def sample_bigram_pairs(
    n: int,
    probs: np.ndarray,
    pi: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    v = probs.shape[0]
    x = rng.choice(v, size=n, p=pi).astype(np.int64)
    y = np.empty(n, dtype=np.int64)
    for i, tok in enumerate(x):
        y[i] = rng.choice(v, p=probs[int(tok)])
    return x[:, None], y


def sample_bigram_lm_windows(
    n: int,
    context_length: int,
    probs: np.ndarray,
    pi: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    v = probs.shape[0]
    L = int(context_length)
    seq = np.empty((n, L + 1), dtype=np.int64)
    for i in range(n):
        s = [int(rng.choice(v, p=pi))]
        while len(s) < L + 1:
            s.append(int(rng.choice(v, p=probs[s[-1]])))
        seq[i] = np.asarray(s, dtype=np.int64)
    x = seq[:, :-1].copy()
    y = seq[:, 1:].copy()
    return x, y


def make_bigram_dataset(
    *,
    vocab_size: int,
    context_length: int,
    train_size: int,
    test_size: int,
    seed: int,
    table_seed: int,
    alpha: float = 1.0,
    layout: str = "lm",
) -> dict[str, np.ndarray]:
    table_rng = np.random.default_rng(table_seed)
    probs = make_bigram_transition(vocab_size, rng=table_rng, alpha=alpha)
    pi = stationary_distribution(probs)

    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)

    if layout == "pairs":
        x_train, y_train = sample_bigram_pairs(train_size, probs, pi, rng_train)
        x_test, y_test = sample_bigram_pairs(test_size, probs, pi, rng_test)
    elif layout == "lm":
        x_train, y_train = sample_bigram_lm_windows(
            train_size, context_length, probs, pi, rng_train
        )
        x_test, y_test = sample_bigram_lm_windows(
            test_size, context_length, probs, pi, rng_test
        )
    else:
        raise ValueError("layout must be 'lm' or 'pairs'")

    return {
        "probs": probs,
        "pi": pi,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
