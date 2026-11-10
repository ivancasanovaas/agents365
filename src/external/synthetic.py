"""Synthetic toy match, ONLY to smoke-test the bench plumbing. Results on it say nothing about football."""
from __future__ import annotations

import numpy as np

from .common import A, D, FPS, H, PITCH_M, X, Tracking


def make_match(seed: int, minutes: float = 10.0, n_per_team: int = 11):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FPS)
    P = 2 * n_per_team
    team = np.repeat(np.array([0, 1], dtype=np.int8), n_per_team)

    base = rng.uniform([0, 0], PITCH_M, size=(P, 2))
    walk = np.cumsum(rng.normal(0, 0.15, size=(n, P, 2)), axis=0)
    players = np.clip(base + walk - walk.mean(0), 0, PITCH_M)

    codes = np.full(n, D, dtype=np.int8)
    t = 0
    owner_team = int(rng.integers(0, 2))
    while t < n:
        length = int(rng.integers(5, 25) * FPS)
        codes[t:t + length] = H if owner_team == 0 else A
        t += length
        if rng.random() < 0.25 and t < n:  # a stoppage
            stop = int(rng.integers(3, 10) * FPS)
            codes[t:t + stop] = D
            t += stop
        owner_team = 1 - owner_team

    ball = np.full((n, 2), np.nan)
    for f in range(n):
        if codes[f] == D:  # dead ball: stays where it was (or mid-pitch at the start)
            ball[f] = ball[f - 1] if f else (PITCH_M[0] / 2, PITCH_M[1] / 2)
            continue
        members = np.flatnonzero(team == (0 if codes[f] == H else 1))
        carrier = members[(f // (2 * FPS)) % len(members)]  # carrier changes every 2 s (a "pass")
        ball[f] = players[f, carrier] + rng.normal(0, 0.3, 2)
    # carrier is always the nearest player on average: good enough for plumbing tests.
    gt = codes.copy()
    gt[: 5 * FPS] = X  # pretend the first seconds are not graded
    return Tracking(ball=ball, players=players, slot_team=team), gt
