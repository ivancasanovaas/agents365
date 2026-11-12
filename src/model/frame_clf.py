"""Discriminative per-frame emissions for the possession HMM (EXP-010): softmax regression, numpy only.

Features come from tracking alone (no labels) via `frame_features.build_frame_table`; the classifier is trained on EXTERNAL
labelled games only. Home/away symmetry is enforced by training on the team-swapped copy of every frame as well.
The classifier's posterior is turned into a scaled likelihood P(state | x) / P(state) and decoded with the same transition
matrix and Viterbi as the naive-Bayes model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import features as F
from .hmm import CROWD_EDGES, N_STATES, OUT_RUN_EDGES_S, viterbi

L2 = 1e-2
ITERS = 300
LR = 0.05
TRAIN_STRIDE = 5          # every 5th graded frame (frames are strongly autocorrelated)


def _names() -> List[str]:
    names = ["ball_vis"]
    for t in ("h", "a"):
        names += ["dmin_" + t, "d3_" + t, "dcen_" + t, "near5_" + t, "near10_" + t]
    names += ["bspeed", "bspeed_ok", "out"] + ["out_b{}".format(k) for k in range(len(OUT_RUN_EDGES_S) + 1)]
    names += ["crowd_log", "crowd_ok"] + ["crowd_b{}".format(k) for k in range(len(CROWD_EDGES) + 1)]
    names += ["dpair", "spair", "dsx", "dsy", "dspd", "sspd", "team_ok"]
    return names


FEATURE_NAMES = _names()


def _zero(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(a)
    return np.where(ok, a, 0.0), ok


def feature_matrix(df: pd.DataFrame, out_run_s: np.ndarray, swap: bool = False) -> np.ndarray:
    """(n, d) float32. `swap=True` exchanges the home and away roles (training-time symmetry augmentation)."""
    def col(base, team):
        t = {"h": "a", "a": "h"}[team] if swap else team
        return df["{}_{}".format(base, t)].to_numpy(dtype=float)

    n = len(df)
    cols = {}
    vis = df["ball_visible"].to_numpy(dtype=float)
    cols["ball_vis"] = vis
    for t in ("h", "a"):
        for base, name in (("dball_min", "dmin_"), ("dball_3", "d3_"), ("dball_cen", "dcen_")):
            v, _ = _zero(col(base, t))
            cols[name + t] = np.log1p(v)
        for base, name in (("near5", "near5_"), ("near10", "near10_")):
            v, _ = _zero(col(base, t))
            cols[name + t] = v
    bs, bok = _zero(df["ball_speed"].to_numpy(dtype=float))
    cols["bspeed"], cols["bspeed_ok"] = np.log1p(bs), bok.astype(float)
    out = df["ball_out"].to_numpy(dtype=float)
    cols["out"] = out
    bucket = np.digitize(out_run_s, OUT_RUN_EDGES_S)
    for k in range(len(OUT_RUN_EDGES_S) + 1):
        cols["out_b{}".format(k)] = ((out > 0) & (bucket == k)).astype(float)
    cs, cok = _zero(df["crowd_speed"].to_numpy(dtype=float))
    cols["crowd_log"], cols["crowd_ok"] = np.log1p(cs), cok.astype(float)
    cb = np.digitize(cs, CROWD_EDGES)
    for k in range(len(CROWD_EDGES) + 1):
        cols["crowd_b{}".format(k)] = (cok & (cb == k)).astype(float)

    ph, pa = col("pair", "h"), col("pair", "a")
    sh, sa = col("sx", "h"), col("sx", "a")
    yh, ya = col("sy", "h"), col("sy", "a")
    vh, va = col("spd_mean", "h"), col("spd_mean", "a")
    ok = np.isfinite(ph) & np.isfinite(pa)
    z = lambda a: np.where(np.isfinite(a), a, 0.0)  # noqa: E731
    cols["dpair"] = np.where(ok, np.clip(z(ph) - z(pa), -15, 15) / 10.0, 0.0)
    cols["spair"] = np.where(ok, (z(ph) + z(pa)) / 40.0, 0.0)
    cols["dsx"] = np.where(ok, np.clip(z(sh) - z(sa), -15, 15) / 10.0, 0.0)
    cols["dsy"] = np.where(ok, np.clip(z(yh) - z(ya), -10, 10) / 10.0, 0.0)
    cols["dspd"] = np.where(ok, np.clip(z(vh) - z(va), -3, 3) / 3.0, 0.0)
    cols["sspd"] = np.where(ok, (z(vh) + z(va)) / 3.0, 0.0)
    cols["team_ok"] = ok.astype(float)
    X = np.stack([cols[name] for name in FEATURE_NAMES], axis=1)
    assert X.shape == (n, len(FEATURE_NAMES))
    return X.astype(np.float32)


def build_features(tr, runs, swap_too: bool = False):
    """Frame table -> (X, X_swapped or None). Imported lazily to keep the module light."""
    from .frame_features import build_frame_table
    df = build_frame_table(tr, runs)
    out_run_s = F.run_length_per_frame(F.outside_mask(tr.ball)) / 25.0
    X = feature_matrix(df, out_run_s, swap=False)
    return X, (feature_matrix(df, out_run_s, swap=True) if swap_too else None)


@dataclass
class SoftmaxModel:
    W: np.ndarray          # (d + 1, 3), last row = bias
    mean: np.ndarray
    std: np.ndarray
    log_prior: np.ndarray  # (3,) class prior of the training set


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = ITERS, lr: float = LR) -> SoftmaxModel:
    """Multinomial logistic regression with L2 on standardised features; Adam, full batch, deterministic."""
    X = X.astype(np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], axis=1)
    Y = np.eye(N_STATES)[y]
    W = np.zeros((Z.shape[1], N_STATES))
    m, v = np.zeros_like(W), np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        P = _softmax(Z @ W)
        G = Z.T @ (P - Y) / len(Z)
        G[:-1] += l2 * W[:-1]
        m = b1 * m + (1 - b1) * G
        v = b2 * v + (1 - b2) * G * G
        W -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    prior = np.bincount(y, minlength=N_STATES).astype(float)
    return SoftmaxModel(W, mean, std, np.log(prior / prior.sum()))


def predict_log_proba(model: SoftmaxModel, X: np.ndarray) -> np.ndarray:
    Z = np.concatenate([(X.astype(np.float64) - model.mean) / model.std, np.ones((len(X), 1))], axis=1)
    logits = Z @ model.W
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def training_set(Xs: Sequence[np.ndarray], Xs_swapped: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                 stride: int = TRAIN_STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Graded frames with label H/A/D of several games, every `stride`-th, plus the team-swapped copy (H<->A)."""
    xs, ys = [], []
    swap = np.array([1, 0, 2])
    for X, Xs_, gt in zip(Xs, Xs_swapped, gts):
        idx = np.flatnonzero(gt < N_STATES)[::stride]
        xs += [X[idx], Xs_[idx]]
        ys += [gt[idx].astype(np.int64), swap[gt[idx]].astype(np.int64)]
    return np.concatenate(xs), np.concatenate(ys)


def decode_softmax(model: SoftmaxModel, X: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                   runs: Sequence[Tuple[int, int]], temp: float, fill: int = 3) -> np.ndarray:
    ll = temp * (predict_log_proba(model, X) - model.log_prior[None, :])
    out = np.full(len(X), fill, dtype=np.int8)
    for lo, hi in runs:
        out[lo:hi] = viterbi(ll[lo:hi], log_trans, log_init)
    return out
