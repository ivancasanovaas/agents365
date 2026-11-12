"""Three-state (H, A, D) HMM with Viterbi decoding over per-frame tracking evidence.

Pure numpy, cross-platform, deterministic. States use the possession_metrics codes: H=0, A=1, D=2.

Evidence per frame (naive Bayes, optionally tempered by `temp` because consecutive frames are
strongly correlated):
  - team symbol: (team of nearest player to the ball, distance bucket); absent when the ball is hidden
  - ball speed bucket over +-LAG frames; absent when the ball is hidden at either end
  - OPTIONAL extra dead-ball streams (EXP-003), each a small categorical symbol per frame:
      "out"      ball outside the pitch rectangle: 0 inside, 1..4 = length bucket of the outside run
      "crowd"    median player speed over a ~1 s window, 5 buckets
      "gapedge"  while the ball is missing: bucket of the last visible ball's distance to a line
Frames without evidence add nothing, so H/A are carried by the transition prior across gaps.

Parameters are ESTIMATED from labelled frames of OTHER games (see PROVENANCE.md); nothing here is fitted
to the competition match or its public ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..external.baseline import nearest_player
from ..external.common import FPS, PITCH_M, Tracking
from . import features as F

N_STATES = 3
DIST_EDGES = np.array([1.0, 2.0, 4.0, 8.0])         # m: 5 buckets (assumption: coarse, see PROVENANCE)
SPEED_EDGES = np.array([0.5, 2.0, 6.0, 12.0])        # m/s: 5 buckets (assumption)
N_TEAM_SYM = 2 * (len(DIST_EDGES) + 1)
N_SPEED = len(SPEED_EDGES) + 1
LAG = 5                                              # frames each side for the speed estimate

# EXP-003 buckets: coarse ASSUMPTIONS, never tuned (PROVENANCE.md). Learned weights come from external games.
OUT_RUN_EDGES_S = np.array([0.5, 2.0, 6.0])          # outside-run length: <0.5 s, 0.5-2, 2-6, >6 s
CROWD_EDGES = np.array([0.4, 0.8, 1.2, 1.8])         # m/s over a ~1 s window: 5 buckets
GAP_EDGE_EDGES = np.array([0.0, 2.0, 6.0, 15.0])     # m, signed: outside, 0-2, 2-6, 6-15, >15
EXTRA_NSYM = {
    "out": len(OUT_RUN_EDGES_S) + 2,
    "crowd": len(CROWD_EDGES) + 1,
    "gapedge": len(GAP_EDGE_EDGES) + 1,
}
ALL_EXTRA = tuple(EXTRA_NSYM)


@dataclass
class Evidence:
    team_sym: np.ndarray   # (n,) int, -1 = none
    speed_sym: np.ndarray  # (n,) int, -1 = none
    extra: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (n,) int, -1 = none


@dataclass
class HmmParams:
    log_trans: np.ndarray   # (3,3)
    log_team: np.ndarray    # (3, N_TEAM_SYM)
    log_speed: np.ndarray   # (3, N_SPEED)
    log_prior: np.ndarray   # (3,)
    log_extra: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (3, EXTRA_NSYM[name])


def extract_extra(tr: Tracking, streams: Sequence[str], fps: int = FPS,
                  pitch: Tuple[float, float] = PITCH_M) -> Dict[str, np.ndarray]:
    n = tr.n_frames
    out: Dict[str, np.ndarray] = {}
    for name in streams:
        if name not in EXTRA_NSYM:
            raise ValueError("unknown stream {!r}".format(name))
        sym = np.full(n, -1, dtype=np.int64)
        if name == "out":
            vis = np.isfinite(tr.ball[:, 0])
            outside = F.outside_mask(tr.ball, pitch)
            sym[vis] = 0
            run_s = F.run_length_per_frame(outside) / float(fps)
            sym[outside] = 1 + np.digitize(run_s[outside], OUT_RUN_EDGES_S)
        elif name == "crowd":
            speed = F.crowd_speed(tr.players, fps=fps)
            ok = np.isfinite(speed)
            sym[ok] = np.digitize(speed[ok], CROWD_EDGES)
        elif name == "gapedge":
            dist = F.gap_edge_distance(tr.ball, pitch)
            ok = np.isfinite(dist)
            sym[ok] = np.digitize(dist[ok], GAP_EDGE_EDGES)
        out[name] = sym
    return out


def extract_evidence(tr: Tracking, fps: int = FPS, streams: Sequence[str] = ()) -> Evidence:
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
    return Evidence(team_sym, speed_sym, extract_extra(tr, streams, fps) if streams else {})


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

    log_extra = {}
    for name, sym in evidence.extra.items():
        counts = np.zeros((N_STATES, EXTRA_NSYM[name]))
        sel = use & (sym >= 0)
        np.add.at(counts, (lab[sel], sym[sel]), 1)
        log_extra[name] = _norm_log(counts)

    prior = np.bincount(lab[use], minlength=N_STATES).astype(float)
    return HmmParams(log_trans, _norm_log(team), _norm_log(speed), _norm_log(prior), log_extra)


def _loglik(evidence: Evidence, params: HmmParams, temp: float, use_speed: bool,
            streams: Optional[Sequence[str]] = None) -> np.ndarray:
    """`streams`: which extra streams to use. None = every stream present in both evidence and params."""
    n = len(evidence.team_sym)
    ll = np.zeros((n, N_STATES))
    seen = evidence.team_sym >= 0
    ll[seen] += params.log_team[:, evidence.team_sym[seen]].T
    if use_speed:
        seen = evidence.speed_sym >= 0
        ll[seen] += params.log_speed[:, evidence.speed_sym[seen]].T
    names = list(params.log_extra) if streams is None else list(streams)
    for name in names:
        if name in evidence.extra and name in params.log_extra:
            sym = evidence.extra[name]
            seen = sym >= 0
            ll[seen] += params.log_extra[name][:, sym[seen]].T
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
           temp: float = 1.0, use_speed: bool = True, fill: int = 3,
           streams: Optional[Sequence[str]] = None) -> np.ndarray:
    """Viterbi inside each [start, stop) run (never across); frames outside runs get `fill` (X=3)."""
    ll = _loglik(evidence, params, temp, use_speed, streams)
    out = np.full(len(ll), fill, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(ll[start:stop], params.log_trans, params.log_prior)
    return out
