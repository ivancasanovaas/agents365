"""Degrade clean tracking so it resembles the automatic, uncorrected competition tracking.

Pure function of (clean tracking, parameters, seed). Defaults use only numbers published in
DATA.md/CHALLENGE.md (ball missing 41%, ~20.5 of 22 objects present). `mean_gap_frames` and
`p_switch_per_min` are ASSUMPTIONS (not published): keep them as sweep parameters, record in PROVENANCE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import Tracking


@dataclass(frozen=True)
class DegradeParams:
    p_ball_missing: float = 0.41       # published (DATA.md). TOTAL target, counting gaps already in the input
    mean_gap_frames: float = 50.0      # ASSUMPTION: mean length of a ball-missing run (2 s)
    drop_rate: float = 0.068           # published: ~20.5 of 22 objects present on average
    mean_drop_frames: float = 40.0     # ASSUMPTION: mean length of a player dropout
    p_switch_per_min: float = 0.5      # ASSUMPTION: same-team ID swaps per minute
    switch_radius_m: float = 5.0       # ASSUMPTION
    pos_noise_m: float = 0.3           # ASSUMPTION
    fps: int = 25


def _markov_mask(rng, n, p_missing, mean_gap):
    """Boolean mask, True = missing, alternating geometric runs with the given duty cycle."""
    if p_missing <= 0:
        return np.zeros(n, dtype=bool)
    if p_missing >= 1:
        return np.ones(n, dtype=bool)
    mean_present = mean_gap * (1 - p_missing) / p_missing
    mask = np.zeros(n, dtype=bool)
    t = 0
    missing = rng.random() < p_missing
    while t < n:
        mean = mean_gap if missing else mean_present
        length = int(rng.geometric(1.0 / max(mean, 1.0)))
        if missing:
            mask[t:t + length] = True
        t += length
        missing = not missing
    return mask


def degrade(clean: Tracking, seed: int, params: DegradeParams = DegradeParams()) -> Tracking:
    rng = np.random.default_rng(seed)
    n, n_slots, _ = clean.players.shape
    ball = clean.ball.copy()
    players = clean.players.copy()
    team = clean.slot_team.copy()

    # 1. same-team identity swaps: from frame t on, two nearby same-team slots exchange tracks.
    minutes = n / params.fps / 60.0
    n_switch = rng.poisson(params.p_switch_per_min * minutes)
    for t in np.sort(rng.integers(0, n, size=n_switch)):
        i = int(rng.integers(0, n_slots))
        d = np.linalg.norm(players[t] - players[t, i], axis=1)
        d[(team != team[i]) | (np.arange(n_slots) == i)] = np.inf
        d[np.isnan(d)] = np.inf
        j = int(np.argmin(d))
        if d[j] <= params.switch_radius_m:
            players[t:, [i, j]] = players[t:, [j, i]]

    # 2. position noise, then dropouts and ball gaps.
    players += rng.normal(0.0, params.pos_noise_m, players.shape)
    ball += rng.normal(0.0, params.pos_noise_m, ball.shape)
    for s in range(n_slots):
        players[_markov_mask(rng, n, params.drop_rate, params.mean_drop_frames), s] = np.nan
    # Ball: the input may already miss frames. Add gaps so that the TOTAL missing rate hits the
    # target, assuming the new gaps are independent of the existing ones.
    already = float(np.isnan(clean.ball[:, 0]).mean())
    extra = 0.0 if params.p_ball_missing <= already else (params.p_ball_missing - already) / (1.0 - already)
    ball[_markov_mask(rng, n, extra, params.mean_gap_frames)] = np.nan

    # 3. anonymise: shuffle slot order (team array follows the slots).
    perm = rng.permutation(n_slots)
    return Tracking(ball=ball, players=players[:, perm], slot_team=team[perm])
