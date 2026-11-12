"""Frame-level collective descriptors of a match: one row per video frame, a few dozen numeric columns.

Why collective: the 46 track slots are anonymous and swap identities, so only statistics that do not depend on
which slot is which are meaningful (centroids, spreads, speed distributions, compactness, distances to the ball).
Everything is computed from tracking alone (no labels), is deterministic, and is cross-platform numpy/pandas.

The table is a DERIVED FRAME-LEVEL EXPORT of the match: keep it outside the repository (MTAH_DATA_DIR/derived),
never commit it, never print rows of it. Use `aggregate_table` for anything that is shown.

COLUMN_DOC is the single source of truth for what each column means; the notebook prints it as a data
dictionary and a test checks that every column is documented.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..external.common import FPS, PITCH_M, Tracking
from .features import CROWD_LAG, outside_mask

TEAMS = (("h", 0, "home (Barcelona)"), ("a", 1, "away (Athletic Club)"))
MIN_PLAYERS = 3                  # a team statistic needs at least this many visible slots (assumption: bare minimum)
NEAR_RADII_M = (5.0, 10.0)       # reporting radii for "players close to the ball" (descriptive, not thresholds)
PAIR_CHUNK = 4000                # frames per block when computing pairwise distances (memory bound)

_TEAM_DOC = {
    "n": "Visible track slots of the team in this frame (count; slots can be duplicated/missing, ~11 expected).",
    "cx": "Team centroid x (m): mean x of its visible slots along the pitch length (0 = left goal line, 105.3 = right).",
    "cy": "Team centroid y (m): mean y of its visible slots across the pitch width (0..68).",
    "sx": "Team spread in x (m): standard deviation of x over visible slots (depth of the team block).",
    "sy": "Team spread in y (m): standard deviation of y over visible slots (width of the team block).",
    "pair": "Compactness (m): mean pairwise distance between visible slots of the team.",
    "spd_mean": "Mean speed (m/s) of the team's visible slots over a +-1 s window (net displacement / time).",
    "spd_std": "Standard deviation of those speeds across the team's slots (m/s).",
    "spd_med": "Median of those speeds across the team's slots (m/s); robust to an identity swap.",
    "cvx": "Team centroid velocity along x (m/s) over the +-1 s window; sign depends on attack direction (see *_n columns).",
    "cvy": "Team centroid velocity along y (m/s) over the +-1 s window.",
    "dball_min": "Distance (m) from the ball to the nearest visible slot of the team; NaN if the ball is hidden.",
    "dball_3": "Mean distance (m) from the ball to the 3 nearest visible slots of the team; NaN if the ball is hidden.",
    "dball_cen": "Distance (m) from the ball to the team centroid; NaN if the ball is hidden.",
    "near5": "Number of visible slots of the team within 5 m of the ball; NaN if the ball is hidden.",
    "near10": "Number of visible slots of the team within 10 m of the ball; NaN if the ball is hidden.",
}

GLOBAL_DOC = {
    "frame": "Video frame index 0..199318 of the tactical video (row i of a submission).",
    "half": "0 = not graded (pre-match, half-time, post-match); 1 = first graded half; 2 = second graded half.",
    "ball_x": "Ball x (m); NaN when the tracker does not see the ball (about 41% of graded frames).",
    "ball_y": "Ball y (m); NaN when the ball is hidden.",
    "ball_visible": "1 if the ball is tracked in this frame, else 0.",
    "ball_out": "1 if the ball is visible and outside the pitch rectangle 105.3 x 68 m (FIFA Law 9 cue), else 0.",
    "ball_speed": "Ball speed (m/s) over a +-1 s window; NaN if the ball is missing at either end of the window.",
    "n_all": "Visible track slots of both teams (count).",
    "crowd_speed": "Median speed (m/s) over all visible slots over the +-1 s window (collective stillness cue).",
}

COLUMN_DOC: Dict[str, str] = dict(GLOBAL_DOC)
for _t, _code, _name in TEAMS:
    for _k, _v in _TEAM_DOC.items():
        COLUMN_DOC["{}_{}".format(_k, _t)] = "[{}] {}".format(_name, _v)
NORMALISED_DOC = {
    "cx_h_n": "Home centroid x in the NORMALISED frame (home always attacks towards +x), m.",
    "cx_a_n": "Away centroid x in the normalised frame (away attacks towards -x), m.",
    "cvx_h_n": "Home centroid x-velocity in the normalised frame (positive = advancing), m/s.",
    "cvx_a_n": "Away centroid x-velocity in the normalised frame (negative = advancing), m/s.",
    "ball_x_n": "Ball x in the normalised frame, m.",
    "home_attacks_right": "1 if, in this frame's half, the home team attacks towards +x in the raw coordinates, else 0.",
}


def _stack_team(players: np.ndarray, slot_team: np.ndarray, code: int) -> np.ndarray:
    return players[:, np.flatnonzero(slot_team == code)]


def _mean_pairwise(p: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Mean pairwise distance among visible slots, chunked over frames. p: (n,k,2), seen: (n,k)."""
    n, k, _ = p.shape
    out = np.full(n, np.nan)
    eye = np.eye(k, dtype=bool)[None]
    for lo in range(0, n, PAIR_CHUNK):
        hi = min(n, lo + PAIR_CHUNK)
        blk = np.where(seen[lo:hi, :, None], p[lo:hi], 0.0)
        diff = blk[:, :, None, :] - blk[:, None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        ok = seen[lo:hi, :, None] & seen[lo:hi, None, :] & ~eye
        cnt = ok.sum(axis=(1, 2))
        total = (dist * ok).sum(axis=(1, 2))
        res = np.full(hi - lo, np.nan)
        good = cnt > 0
        res[good] = total[good] / cnt[good]
        out[lo:hi] = res
    return out


def _windowed(a: np.ndarray, lag: int, fn) -> np.ndarray:
    """Apply fn(a[t+lag], a[t-lag]) for the interior frames; NaN at the borders."""
    if len(a) <= 2 * lag:
        probe = fn(a[:1], a[:1])
        return np.full((len(a),) + probe.shape[1:], np.nan)
    res = fn(a[2 * lag:], a[:-2 * lag])
    out = np.full((len(a),) + res.shape[1:], np.nan)
    out[lag:len(a) - lag] = res
    return out


def build_frame_table(tr: Tracking, runs: Optional[Sequence[Tuple[int, int]]] = None, fps: int = FPS,
                      lag: int = CROWD_LAG, pitch: Tuple[float, float] = PITCH_M) -> pd.DataFrame:
    n = tr.n_frames
    dt = 2.0 * lag / fps
    ball = tr.ball
    pos = tr.players
    vis = np.isfinite(ball[:, 0])
    cols: Dict[str, np.ndarray] = {"frame": np.arange(n, dtype=np.int64)}

    half = np.zeros(n, dtype=np.int8)
    for i, (lo, hi) in enumerate(runs or [(0, n)]):
        half[lo:hi] = i + 1
    cols["half"] = half
    cols["ball_x"], cols["ball_y"] = ball[:, 0], ball[:, 1]
    cols["ball_visible"] = vis.astype(np.int8)
    cols["ball_out"] = outside_mask(ball, pitch).astype(np.int8)
    cols["ball_speed"] = _windowed(ball, lag, lambda a, b: np.linalg.norm(a - b, axis=1) / dt)

    spd = _windowed(pos, lag, lambda a, b: np.linalg.norm(a - b, axis=2) / dt)       # (n, P)
    seen_all = np.isfinite(pos[:, :, 0])
    cols["n_all"] = seen_all.sum(axis=1).astype(np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cols["crowd_speed"] = np.nanmedian(spd, axis=1)

        for name, code, _ in TEAMS:
            p = _stack_team(pos, tr.slot_team, code)
            s = _stack_team(spd, tr.slot_team, code) if spd.ndim == 2 else spd
            seen = np.isfinite(p[:, :, 0])
            cnt = seen.sum(axis=1)
            ok = cnt >= MIN_PLAYERS
            cols["n_" + name] = cnt.astype(np.int16)

            def keep(a):
                a = np.asarray(a, dtype=float)
                a[~ok] = np.nan
                return a

            cx, cy = keep(np.nanmean(p[:, :, 0], axis=1)), keep(np.nanmean(p[:, :, 1], axis=1))
            cols["cx_" + name], cols["cy_" + name] = cx, cy
            cols["sx_" + name], cols["sy_" + name] = keep(np.nanstd(p[:, :, 0], axis=1)), keep(np.nanstd(p[:, :, 1], axis=1))
            cols["pair_" + name] = keep(_mean_pairwise(p, seen))
            cols["spd_mean_" + name] = keep(np.nanmean(s, axis=1))
            cols["spd_std_" + name] = keep(np.nanstd(s, axis=1))
            cols["spd_med_" + name] = keep(np.nanmedian(s, axis=1))
            cen = np.stack([cx, cy], axis=1)
            cols["cvx_" + name] = _windowed(cx, lag, lambda a, b: (a - b) / dt)
            cols["cvy_" + name] = _windowed(cy, lag, lambda a, b: (a - b) / dt)

            d = np.sqrt(((p - ball[:, None, :]) ** 2).sum(-1))                      # NaN where slot or ball missing
            has = np.isfinite(d).any(axis=1)
            srt = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
            dmin = np.where(has, srt[:, 0], np.nan)
            k3 = min(3, srt.shape[1])
            top3 = srt[:, :k3]
            d3 = np.where(np.isfinite(top3).all(axis=1), top3.mean(axis=1), np.nan)
            cols["dball_min_" + name] = dmin
            cols["dball_3_" + name] = d3
            cols["dball_cen_" + name] = keep(np.linalg.norm(cen - ball, axis=1))
            for r in NEAR_RADII_M:
                c = np.where(has, (np.where(np.isfinite(d), d, np.inf) < r).sum(axis=1).astype(float), np.nan)
                cols["near{}_{}".format(int(r), name)] = c

    df = pd.DataFrame(cols)
    return df[list(COLUMN_DOC)]


def attack_direction(df: pd.DataFrame) -> Dict[int, int]:
    """+1 if the home team attacks towards +x in that half, else -1. Unsupervised: the home team defends the side
    where its centroid sits on average relative to the away team (each team's block is on its own side of the
    other's: the two goalkeepers and back lines anchor them). Uses graded frames only."""
    out = {}
    for h in sorted(set(df["half"].unique()) - {0}):
        sub = df[df["half"] == h]
        ch, ca = sub["cx_h"].median(), sub["cx_a"].median()
        out[int(h)] = 1 if ch < ca else -1
    return out


def add_normalised(df: pd.DataFrame, pitch: Tuple[float, float] = PITCH_M) -> pd.DataFrame:
    """Add *_n columns so that the home team always attacks towards +x (flip the half where it does not)."""
    length = pitch[0]
    dirs = attack_direction(df)
    right = df["half"].map(lambda h: 1 if dirs.get(int(h), 1) == 1 else 0).astype(np.int8)
    flip = right == 0
    out = df.copy()
    out["home_attacks_right"] = right
    out["cx_h_n"] = np.where(flip, length - df["cx_h"], df["cx_h"])
    out["cx_a_n"] = np.where(flip, length - df["cx_a"], df["cx_a"])
    out["cvx_h_n"] = np.where(flip, -df["cvx_h"], df["cvx_h"])
    out["cvx_a_n"] = np.where(flip, -df["cvx_a"], df["cvx_a"])
    out["ball_x_n"] = np.where(flip, length - df["ball_x"], df["ball_x"])
    return out


FULL_DOC: Dict[str, str] = {**COLUMN_DOC, **NORMALISED_DOC}


def aggregate_table(df: pd.DataFrame, columns: Sequence[str], mask: Optional[np.ndarray] = None,
                    qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """Per-column aggregates (count, null share, mean, std, quantiles). Never contains a row of data."""
    sub = df if mask is None else df[mask]
    rows = []
    for c in columns:
        v = sub[c].to_numpy(dtype=float)
        good = v[np.isfinite(v)]
        row = {"column": c, "n": int(good.size), "null_share": float(1 - good.size / max(len(v), 1)),
               "mean": float(good.mean()) if good.size else np.nan, "std": float(good.std()) if good.size else np.nan}
        for q in qs:
            row["p{:02d}".format(int(q * 100))] = float(np.quantile(good, q)) if good.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")
