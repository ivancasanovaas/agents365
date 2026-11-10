"""Three-state (H, A, D) HMM with Viterbi decoding over per-frame tracking evidence.

Pure numpy, cross-platform, deterministic. States use the possession_metrics codes: H=0, A=1, D=2.

Evidence per frame (naive Bayes, optionally tempered by `temp` because consecutive frames are
strongly correlated):
  - team symbol: (team of nearest player to the ball, distance bucket); absent when the ball is hidden
  - ball speed bucket over +-LAG frames; absent when the ball is hidden at either end
Frames without evidence add nothing, so H/A are carried by the transition prior across gaps.

Parameters are ESTIMATED from labelled frames of OTHER games (see PROVENANCE.md); nothing here is fitted
to the competition match or its public ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from ..external.baseline import nearest_player
from ..external.common import FPS, Tracking

N_STATES = 3
DIST_EDGES = np.array([1.0, 2.0, 4.0, 8.0])         # m: 5 buckets (assumption: coarse, see PROVENANCE)
SPEED_EDGES = np.array([0.5, 2.0, 6.0, 12.0])        # m/s: 5 buckets (assumption)
N_TEAM_SYM = 2 * (len(DIST_EDGES) + 1)
N_SPEED = len(SPEED_EDGES) + 1
LAG = 5                                              # frames each side for the speed estimate


@dataclass
class Evidence:
    team_sym: np.ndarray   # (n,) int, -1 = none
    speed_sym: np.ndarray  # (n,) int, -1 = none


@dataclass
class HmmParams:
    log_trans: np.ndarray   # (3,3)
    log_team: np.ndarray    # (3, N_TEAM_SYM)
    log_speed: np.ndarray   # (3, N_SPEED)
    log_prior: np.ndarray   # (3,)


def extract_evidence(tr: Tracking, fps: int = FPS) -> Evidence:
    codes, dist = nearest_player(tr)
    team_sym = np.full(len(codes), -1, dtype=np.int64)
    seen = codes >= 0
    team_sym[seen] = codes[seen] * (len(DIST_EDGES) + 1) + np.digitize(dist[seen], DIST_EDGES)

    ball = tr.ball
    speed_sym = np.full(len(ball), -1, dtype=np.int64)
    if len(ball) > 2 * LAG:
        disp = np.linalg.norm(ball[2 * LAG:] - ball[:-2 * LAG], axis=1)
        speed = disp / (2 * LAG / fps)
        ok = np.isfinite(speed)
        idx = np.arange(LAG, len(ball) - LAG)
        speed_sym[idx[ok]] = np.digitize(speed[ok], SPEED_EDGES)
    return Evidence(team_sym, speed_sym)


def _norm_log(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    p = counts + alpha
    return np.log(p / p.sum(axis=-1, keepdims=True))


def fit(evidence: Evidence, labels: np.ndarray, mask: np.ndarray, switch_scale: float = 1.0) -> HmmParams:
    """Supervised estimation on labelled frames (labels in 0..2, mask = frames usable for training)."""
    lab = labels.astype(np.int64)
    use = mask & (lab < N_STATES)

    trans = np.zeros((N_STATES, N_STATES))
    both = use[:-1] & use[1:]
    np.add.at(trans, (lab[:-1][both], lab[1:][both]), 1)
    log_trans = _norm_log(trans, alpha=0.5)
    if switch_scale != 1.0:  # multiply off-diagonal mass, renormalise
        p = np.exp(log_trans)
        off = ~np.eye(N_STATES, dtype=bool)
        p[off] *= switch_scale
        p /= p.sum(axis=1, keepdims=True)
        log_trans = np.log(p)

    team = np.zeros((N_STATES, N_TEAM_SYM))
    sel = use & (evidence.team_sym >= 0)
    np.add.at(team, (lab[sel], evidence.team_sym[sel]), 1)
    speed = np.zeros((N_STATES, N_SPEED))
    sel = use & (evidence.speed_sym >= 0)
    np.add.at(speed, (lab[sel], evidence.speed_sym[sel]), 1)

    prior = np.bincount(lab[use], minlength=N_STATES).astype(float)
    return HmmParams(log_trans, _norm_log(team), _norm_log(speed), _norm_log(prior))


def _loglik(evidence: Evidence, params: HmmParams, temp: float, use_speed: bool) -> np.ndarray:
    n = len(evidence.team_sym)
    ll = np.zeros((n, N_STATES))
    seen = evidence.team_sym >= 0
    ll[seen] += params.log_team[:, evidence.team_sym[seen]].T
    if use_speed:
        seen = evidence.speed_sym >= 0
        ll[seen] += params.log_speed[:, evidence.speed_sym[seen]].T
    return ll * temp


def viterbi(ll: np.ndarray, log_trans: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    n = len(ll)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    delta = log_prior + ll[0]
    back = np.zeros((n, N_STATES), dtype=np.int8)
    for t in range(1, n):
        cand = delta[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + ll[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(evidence: Evidence, params: HmmParams, runs: Iterable[Tuple[int, int]],
           temp: float = 1.0, use_speed: bool = True, fill: int = 3) -> np.ndarray:
    """Viterbi inside each [start, stop) run (never across); frames outside runs get `fill` (X=3)."""
    ll = _loglik(evidence, params, temp, use_speed)
    out = np.full(len(ll), fill, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(ll[start:stop], params.log_trans, params.log_prior)
    return out
