"""Four-state HMM whose dead-ball state remembers who had the ball before it (EXP-015).

States: H=0, A=1, D_H=2 (dead after a home possession), D_A=3 (dead after an away possession). The emission model is the
unchanged 3-class (H, A, D) model: both dead states use the D emission. Output labels are mapped back to H/A/D (X outside runs).
Pure numpy, deterministic. Idea: the PathCRF paper models the ball leaving the pitch with dedicated states and constrained
transitions instead of one memoryless dead state.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

K4 = 4


def to_four_state(codes: np.ndarray, runs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Labels 0..2 (H, A, D) -> 0..3 inside the runs; -1 where not defined (outside the runs, other labels, or D frames before
    the first possession of a half). A D frame takes the team of the last H/A frame before it in the same run."""
    out = np.full(len(codes), -1, dtype=np.int64)
    for lo, hi in runs:
        last = -1
        for t in range(lo, hi):
            c = int(codes[t])
            if c in (0, 1):
                out[t] = c
                last = c
            elif c == 2 and last >= 0:
                out[t] = 2 + last
    return out


def fit_transitions4(codes_list: Sequence[np.ndarray], runs_list: Sequence[Sequence[Tuple[int, int]]],
                     alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """-> (log_trans (4,4), log_prior (4,)) by counting consecutive defined frames of the converted labels."""
    trans = np.zeros((K4, K4))
    prior = np.zeros(K4)
    for codes, runs in zip(codes_list, runs_list):
        lab = to_four_state(codes, runs)
        ok = (lab[:-1] >= 0) & (lab[1:] >= 0)
        np.add.at(trans, (lab[:-1][ok], lab[1:][ok]), 1)
        prior += np.bincount(lab[lab >= 0], minlength=K4)
    p = trans + alpha
    p = p / p.sum(axis=1, keepdims=True)
    q = prior + alpha
    return np.log(p), np.log(q / q.sum())


def viterbi4(ll: np.ndarray, log_trans: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    n = len(ll)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    delta = log_prior + ll[0]
    back = np.zeros((n, K4), dtype=np.int8)
    for t in range(1, n):
        cand = delta[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + ll[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode4(ll3: np.ndarray, log_trans4: np.ndarray, log_prior4: np.ndarray,
            runs: Iterable[Tuple[int, int]], fill: int = 3) -> np.ndarray:
    """`ll3`: (n, 3) per-frame log-likelihoods of H, A, D (already tempered). Returns codes H0 A1 D2, `fill` outside runs."""
    ll4 = np.concatenate([ll3, ll3[:, 2:3]], axis=1)
    out = np.full(len(ll3), fill, dtype=np.int8)
    for lo, hi in runs:
        path = viterbi4(ll4[lo:hi], log_trans4, log_prior4)
        out[lo:hi] = np.where(path >= 2, 2, path)
    return out
