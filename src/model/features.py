"""Physical dead-ball cues computed from tracking. Pure numpy, deterministic, no labels involved.

All inputs are metres/frames of a `Tracking`; the pitch rectangle is the published competition size
(PROVENANCE.md). Bucket edges in hmm.py are coarse ASSUMPTIONS, not tuned (see PROVENANCE.md).
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from ..external.common import FPS, PITCH_M

CROWD_LAG = 12   # frames each side (about 1 s total window): robust to per-frame position noise


def outside_mask(ball: np.ndarray, pitch: Tuple[float, float] = PITCH_M) -> np.ndarray:
    """True where the ball is visible and outside the pitch rectangle (FIFA Law 9: out of play when it
    wholly crosses a line; here the centre point is used)."""
    length, width = pitch
    vis = np.isfinite(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return vis & ((x < 0) | (x > length) | (y < 0) | (y > width))


def run_length_per_frame(mask: np.ndarray) -> np.ndarray:
    """For every frame, the length of the maximal True run it belongs to (0 where mask is False)."""
    out = np.zeros(mask.shape[0], dtype=np.int64)
    if mask.size == 0:
        return out
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
        out[s:e] = e - s
    return out


def crowd_speed(players: np.ndarray, lag: int = CROWD_LAG, fps: int = FPS) -> np.ndarray:
    """Median over visible slots of net displacement per second across a +-lag window (m/s).
    NaN at the borders and where no slot is visible at both ends. Slots appear/vanish/swap, so only a
    collective statistic is meaningful."""
    n = players.shape[0]
    out = np.full(n, np.nan)
    if n <= 2 * lag:
        return out
    disp = np.linalg.norm(players[2 * lag:] - players[:-2 * lag], axis=2) / (2.0 * lag / fps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out[lag:n - lag] = np.nanmedian(disp, axis=1)
    return out


CROWD_QUANTILES = (0.2, 0.4, 0.6, 0.8)   # quintiles: assumption-free equal-share buckets per game


def crowd_edges_for_game(players: np.ndarray, runs, quantiles=CROWD_QUANTILES, lag: int = CROWD_LAG,
                         fps: int = FPS) -> np.ndarray:
    """Per-game bucket edges for the crowd-speed stream: the given quantiles of this game's own crowd speed over its
    graded frames. Removes differences in overall tempo between games (domain shift) without using any label."""
    speed = crowd_speed(players, lag=lag, fps=fps)
    keep = np.zeros(len(speed), dtype=bool)
    for lo, hi in runs:
        keep[lo:hi] = True
    vals = speed[keep & np.isfinite(speed)]
    if vals.size == 0:
        raise ValueError("no finite crowd speed inside the graded runs")
    return np.quantile(vals, quantiles)


def signed_edge_distance(xy: np.ndarray, pitch: Tuple[float, float] = PITCH_M) -> np.ndarray:
    """Signed distance (m) to the nearest touch/goal line; negative = outside."""
    length, width = pitch
    return np.minimum.reduce([xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]])


def gap_edge_distance(ball: np.ndarray, pitch: Tuple[float, float] = PITCH_M) -> np.ndarray:
    """For frames inside a ball-missing gap: signed distance to the nearest line of the LAST visible ball.
    NaN while the ball is visible or before it was ever seen."""
    n = ball.shape[0]
    vis = np.isfinite(ball[:, 0])
    idx = np.where(vis, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.full(n, np.nan)
    sel = (~vis) & (last >= 0)
    out[sel] = signed_edge_distance(ball[last[sel]], pitch)
    return out
