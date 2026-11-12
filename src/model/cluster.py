"""Minimal, deterministic, numpy-only PCA / k-means / adjusted Rand index for exploratory clustering of frames.

No labels are involved. Clusters are NOT labels: mapping a cluster to H/A/D must be learned on EXTERNAL labelled
games, never decided by eye on the competition match (RULES.md, hand annotation).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns (std 0 -> 1). Returns (Z, mean, std). X must have no NaN."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def pca(Z: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """-> (components (k, d), explained variance ratio (k,)) of already-centred data via SVD."""
    _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    var = s ** 2
    return vt[:k], var[:k] / var.sum()


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1.0 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    return np.array(centers)


def assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype=np.int64)
    for lo in range(0, len(X), 20000):
        blk = X[lo:lo + 20000]
        out[lo:lo + 20000] = ((blk[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
    return out


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 3, n_iter: int = 60):
    """-> (labels, centers, inertia) of the best of n_init k-means++ runs. Deterministic for a seed."""
    best = None
    for init in range(n_init):
        rng = np.random.default_rng(seed * 1000 + init)
        centers = _kmeans_pp(X, k, rng)
        for _ in range(n_iter):
            labels = assign(X, centers)
            new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        labels = assign(X, centers)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    return best


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings (1 = identical partitions, ~0 = chance)."""
    a_ids, a_inv = np.unique(a, return_inverse=True)
    b_ids, b_inv = np.unique(b, return_inverse=True)
    table = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    np.add.at(table, (a_inv, b_inv), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_ij = comb(table).sum()
    sum_a, sum_b = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(float(len(a)))
    maximum = 0.5 * (sum_a + sum_b)
    return 1.0 if maximum == expected else float((sum_ij - expected) / (maximum - expected))
