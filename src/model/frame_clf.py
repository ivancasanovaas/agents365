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


WINDOWS_S = (2.0, 5.0)           # +- seconds for the context means (EXP-012)
GAP_AGE_CAP = 1500               # frames (60 s): ages are capped
_CTX_SERIES = ("dpair", "dnear5", "snear", "crowd", "dspd", "out", "vis")
CTX_WIN_NAMES = ["{}_w{}".format(s, int(w)) for w in WINDOWS_S for s in _CTX_SERIES]
CTX_GAP_NAMES = ["s_smooth", "gap_last_s", "gap_last_age", "gap_next_s", "gap_next_age"]
CTX_NAMES = CTX_WIN_NAMES + CTX_GAP_NAMES


def feature_names(context: bool = False) -> List[str]:
    return list(FEATURE_NAMES) + (CTX_NAMES if context else [])


def _window_mean(a: np.ndarray, valid: np.ndarray, half: int) -> np.ndarray:
    """Mean of `a` over the +-half window using only valid entries; 0 where the window has none."""
    n = len(a)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, a, 0.0))))
    ccnt = np.concatenate(([0.0], np.cumsum(valid.astype(float))))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - half), np.minimum(n, idx + half + 1)
    cnt = ccnt[hi] - ccnt[lo]
    out = np.zeros(n)
    ok = cnt > 0
    out[ok] = (csum[hi] - csum[lo])[ok] / cnt[ok]
    return out


def context_matrix(X: np.ndarray, fps: int = 25) -> np.ndarray:
    """Temporal-context features computed from the base feature matrix of ONE full-length game (n, d) -> (n, 19).
    Window means over +-2 s and +-5 s of the team/ball signals, and for ball-missing frames the smoothed 'which team is
    nearer to the ball' evidence at the last and at the next frame with a visible ball, with its age. Non-causal on purpose
    (offline decoding). Antisymmetric signals flip sign under the home/away swap automatically."""
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    col = lambda name: X[:, idx[name]].astype(float)  # noqa: E731
    n = len(X)
    vis = col("ball_vis") > 0
    team_ok = col("team_ok") > 0
    s_raw = np.tanh((np.expm1(col("dmin_a")) - np.expm1(col("dmin_h"))) / 5.0)       # > 0: home nearer
    series = {"dpair": (col("dpair"), team_ok), "dnear5": (col("near5_h") - col("near5_a"), vis),
              "snear": (s_raw, vis), "crowd": (col("crowd_log"), np.ones(n, bool)),
              "dspd": (col("dspd"), team_ok), "out": (col("out"), np.ones(n, bool)),
              "vis": (col("ball_vis"), np.ones(n, bool))}
    cols = []
    for w in WINDOWS_S:
        half = int(round(w * fps))
        cols += [_window_mean(series[s][0], series[s][1], half) for s in _CTX_SERIES]

    half1 = fps                                                                        # +-1 s smoothing of the evidence
    csum = np.concatenate(([0.0], np.cumsum(np.where(vis, s_raw, 0.0))))
    ccnt = np.concatenate(([0.0], np.cumsum(vis.astype(float))))
    ar = np.arange(n)
    lo, hi = np.maximum(0, ar - half1), np.minimum(n, ar + half1 + 1)
    cnt = ccnt[hi] - ccnt[lo]
    s_sm = np.full(n, np.nan)
    s_sm[cnt > 0] = ((csum[hi] - csum[lo])[cnt > 0]) / cnt[cnt > 0]
    finite = np.isfinite(s_sm)
    last = np.maximum.accumulate(np.where(finite, ar, -1))
    nxt = np.minimum.accumulate(np.where(finite, ar, n + GAP_AGE_CAP)[::-1])[::-1]
    gap = (~vis).astype(float)
    have_last, have_next = last >= 0, nxt < n
    last_s = np.where(have_last, s_sm[np.clip(last, 0, n - 1)], 0.0)
    next_s = np.where(have_next, s_sm[np.clip(nxt, 0, n - 1)], 0.0)
    age_last = np.where(have_last, np.minimum(ar - last, GAP_AGE_CAP), GAP_AGE_CAP)
    age_next = np.where(have_next, np.minimum(nxt - ar, GAP_AGE_CAP), GAP_AGE_CAP)
    cols += [np.where(finite, s_sm, 0.0), gap * last_s, gap * np.log1p(age_last / fps),
             gap * next_s, gap * np.log1p(age_next / fps)]
    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape == (n, len(CTX_NAMES))
    return out


def build_features(tr, runs, swap_too: bool = False, context: bool = False):
    """Frame table -> (X, X_swapped or None). With `context`, the temporal-context columns are appended (EXP-012).
    Imported lazily to keep the module light."""
    from .frame_features import build_frame_table
    df = build_frame_table(tr, runs)
    out_run_s = F.run_length_per_frame(F.outside_mask(tr.ball)) / 25.0

    def make(swap):
        X = feature_matrix(df, out_run_s, swap=swap)
        return np.concatenate([X, context_matrix(X)], axis=1) if context else X
    return make(False), (make(True) if swap_too else None)


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


@dataclass
class MlpModel:
    """One hidden layer (ReLU) + softmax output, numpy only (EXP-013). Same interface as SoftmaxModel for decoding."""
    W1: np.ndarray         # (d, h)
    b1: np.ndarray         # (h,)
    W2: np.ndarray         # (h, 3)
    b2: np.ndarray         # (3,)
    mean: np.ndarray
    std: np.ndarray
    log_prior: np.ndarray

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float64) - self.mean) / self.std
        logits = np.maximum(Z @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


MLP_HIDDEN, MLP_L2, MLP_LR, MLP_ITERS = 32, 1e-3, 0.02, 300


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = MLP_HIDDEN, l2: float = MLP_L2, lr: float = MLP_LR,
            iters: int = MLP_ITERS, seed: int = 0) -> MlpModel:
    """Fixed-hyperparameter MLP trained with Adam, full batch, deterministic for a seed."""
    X = X.astype(np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = (X - mean) / std
    Y = np.eye(N_STATES)[y]
    rng = np.random.default_rng(seed)
    d = Z.shape[1]
    params = [rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), np.zeros(hidden),
              rng.normal(0, np.sqrt(2.0 / hidden), (hidden, N_STATES)), np.zeros(N_STATES)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(Z)
    for t in range(1, iters + 1):
        W1, bb1, W2, bb2 = params
        A = np.maximum(Z @ W1 + bb1, 0.0)
        P = _softmax(A @ W2 + bb2)
        dL = (P - Y) / n
        gW2, gb2 = A.T @ dL + l2 * W2, dL.sum(axis=0)
        dA = (dL @ W2.T) * (A > 0)
        gW1, gb1 = Z.T @ dA + l2 * W1, dA.sum(axis=0)
        for k, g in enumerate((gW1, gb1, gW2, gb2)):
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            params[k] = params[k] - lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
    prior = np.bincount(y, minlength=N_STATES).astype(float)
    return MlpModel(params[0], params[1], params[2], params[3], mean, std, np.log(prior / prior.sum()))


def predict_log_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "log_proba"):
        return model.log_proba(X)
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
