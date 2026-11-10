"""Derive per-frame H/A/D codes from event-like input of an EXTERNAL game.

Output codes follow possession_metrics: H=0, A=1, D=2, X=3 (X = not graded / no answer).

Rule (algorithmic, see PROVENANCE.md): possession belongs to the team of the last controlling
event until the opponent registers a controlling event; frames inside a dead-ball interval are D;
frames outside the graded runs are X.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .common import A, D, H, X


def labels_from_events(
    n_frames: int,
    control_events: Sequence[Tuple[int, int]],
    dead_intervals: Iterable[Tuple[int, int]],
    graded_runs: Iterable[Tuple[int, int]],
) -> np.ndarray:
    """control_events: (frame, team 0|1) sorted by frame; dead_intervals/graded_runs: [start, stop)."""
    codes = np.full(n_frames, X, dtype=np.int8)
    events = sorted(control_events)
    for start, stop in graded_runs:
        run = np.full(stop - start, D, dtype=np.int8)
        for i, (frame, team) in enumerate(events):
            nxt = events[i + 1][0] if i + 1 < len(events) else stop
            lo, hi = max(frame, start), min(nxt, stop)
            if lo < hi:
                run[lo - start:hi - start] = H if team == 0 else A
        codes[start:stop] = run
    for start, stop in dead_intervals:
        graded = codes[start:stop] != X
        codes[start:stop][graded] = D
    return codes


def labels_from_ball_state(
    owning_team: np.ndarray, ball_alive: np.ndarray, graded_runs: Iterable[Tuple[int, int]]
) -> np.ndarray:
    """For providers that ship ball state and owning team per frame (verify per dataset first)."""
    n = len(owning_team)
    codes = np.full(n, X, dtype=np.int8)
    for start, stop in graded_runs:
        seg = np.where(owning_team[start:stop] == 0, H, A).astype(np.int8)
        seg[~ball_alive[start:stop].astype(bool)] = D
        codes[start:stop] = seg
    return codes
