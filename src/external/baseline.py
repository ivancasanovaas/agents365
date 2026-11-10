"""Re-implementation of the organiser baseline (nearest player + centred majority vote) with the
EXP-001 switch for frames where nobody has the ball in the window.

`no_ball="D"` reproduces starter/baseline.py::smooth; `no_ball="X"` abstains instead.
Output codes follow possession_metrics: H=0, A=1, D=2, X=3.
"""
from __future__ import annotations

import numpy as np

from .common import A, AWAY, D, H, HOME, NONE, X, Tracking


def nearest_player(tr: Tracking):
    """-> (codes, distance_m): team (0/1) of the nearest visible player to the ball and that distance.
    codes is -1 and distance is NaN when there is no ball or no player."""
    diff = tr.players - tr.ball[:, None, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    dist[np.isnan(dist)] = np.inf
    nearest = dist.argmin(axis=1)
    best = dist[np.arange(len(dist)), nearest]
    codes = tr.slot_team[nearest].astype(np.int8)
    bad = ~np.isfinite(best)
    codes[bad] = NONE
    best[bad] = np.nan
    return codes, best


def nearest_player_codes(tr: Tracking) -> np.ndarray:
    """Per frame: team (0/1) of the nearest visible player to the ball, -1 if no ball/no player."""
    return nearest_player(tr)[0]


def smooth_codes(codes: np.ndarray, window: int, no_ball: str = "D") -> np.ndarray:
    if no_ball not in ("D", "X"):
        raise ValueError("no_ball must be 'D' or 'X'")
    n = len(codes)
    out = np.full(n, D if no_ball == "D" else X, dtype=np.int8)
    if window <= 0:
        out[codes == HOME] = H
        out[codes == AWAY] = A
        return out
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out
