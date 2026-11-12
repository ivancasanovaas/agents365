"""Degrade clean tracking so it resembles the automatic, uncorrected competition tracking.

Pure function of (clean tracking, parameters, seed). Defaults use only numbers published in
DATA.md/CHALLENGE.md (ball missing 41%, ~20.5 of 22 objects present). `mean_gap_frames` and
`p_switch_per_min` are ASSUMPTIONS (not published): keep them as sweep parameters, record in PROVENANCE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import PITCH_M, Tracking


@dataclass(frozen=True)
class DegradeParams:
    p_ball_missing: float = 0.41       # published (DATA.md). TOTAL target, counting gaps already in the input
    mean_gap_frames: float = 50.0      # ASSUMPTION: mean length of a ball-missing run (2 s)
    drop_rate: float = 0.068           # published: ~20.5 of 22 objects present on average
    mean_drop_frames: float = 40.0     # ASSUMPTION: mean length of a player dropout
    p_switch_per_min: float = 0.5      # ASSUMPTION: same-team ID swaps per minute
    switch_radius_m: float = 5.0       # ASSUMPTION
    pos_noise_m: float = 0.3           # ASSUMPTION
    # EXP-003 realism options. Defaults keep the original behaviour (iid noise, no glitches).
    noise_corr: float = 0.0            # AR(1) correlation of position noise (0 = iid, as before)
    p_ball_glitch_per_min: float = 0.0  # spurious ball jumps; estimated from profile aggregates (PROVENANCE.md)
    glitch_max_frames: int = 3         # ASSUMPTION: a glitch lasts 1..3 frames
    glitch_margin_m: float = 10.0      # ASSUMPTION: glitch positions fall within the pitch +- this margin
    fps: int = 25


def calibrated_params(**overrides) -> DegradeParams:
    """Degradation matched to the AGGREGATE profile of the competition tracking (outputs/profile.md), no labels:
    competition ball-step p50/p90/p99 = 5.1/14.3/24.6 m/s, 0.23% of steps > 40 m/s, median-player step speed
    p10/p50/p90 = 0.56/1.34/3.07 m/s. Clean Metrica already sits close to these; iid per-frame noise (the
    original default) inflates them 5-10x. AR(1) noise with rho 0.999 and 0.7 glitches/min reproduce them
    (scripts/calibrate_degrade.py). Ball gaps (bench v2, EXP-008): competition runs have median 10.7 s, p90 33.8 s
    and mean ~17 s, so mean_gap_frames = 400 (geometric: median 11.1 s, p90 36.8 s). `mean_gap_frames=50` is bench v1.
    See PROVENANCE.md."""
    base = dict(noise_corr=0.999, p_ball_glitch_per_min=0.7, mean_gap_frames=400.0)
    base.update(overrides)
    return DegradeParams(**base)


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


def _noise(rng, shape, sigma, rho):
    """iid Gaussian noise, or AR(1) along axis 0 with stationary std `sigma` when rho > 0."""
    if rho <= 0:
        return rng.normal(0.0, sigma, shape)
    eps = rng.normal(0.0, 1.0, shape)
    out = np.empty(shape)
    out[0] = sigma * eps[0]
    k = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + k * eps[t]
    return out


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
    players += _noise(rng, players.shape, params.pos_noise_m, params.noise_corr)
    ball += _noise(rng, ball.shape, params.pos_noise_m, params.noise_corr)
    if params.p_ball_glitch_per_min > 0:
        length, width = PITCH_M
        lo = np.array([-params.glitch_margin_m, -params.glitch_margin_m])
        hi = np.array([length + params.glitch_margin_m, width + params.glitch_margin_m])
        for t in rng.integers(0, n, size=rng.poisson(params.p_ball_glitch_per_min * minutes)):
            ball[t:t + int(rng.integers(1, params.glitch_max_frames + 1))] = rng.uniform(lo, hi)
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
