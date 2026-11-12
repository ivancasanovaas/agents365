"""Adapter for one SkillCorner open-data match (A-League 2024/25, MIT licence). An OPEN game, NOT the
competition match. stdlib + numpy + pandas only.

Source tracking is 10 fps (broadcast-derived, centred metric coordinates). It is resampled to 25 fps:
positions are linearly interpolated only when both neighbouring source frames are valid, else NaN.
Ball: x/y are NaN unless `is_detected` is true (this reproduces broadcast ball gaps). Players keep the
provider's (possibly extrapolated) positions.

Labels (algorithmic, see PROVENANCE.md): inside a period, a frame takes the team of the "phase of
play" covering it (phases_of_play.csv, team_in_possession_id); frames in a period not covered by any
phase are D; frames outside periods are X.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FPS, REPO_ROOT, Tracking

DEFAULT_MATCH = "1874553"
SRC_FPS = 10


def default_dir(match_id: str = DEFAULT_MATCH) -> Path:
    return REPO_ROOT / "ext_open" / "skillcorner_{}".format(match_id)


def _interp(values: np.ndarray, n_out: int) -> np.ndarray:
    """values: (n_src, ...) with NaN. Linear resample to n_out frames at 25 fps, NaN-safe."""
    pos = np.arange(n_out) * (SRC_FPS / FPS)
    lo = np.minimum(np.floor(pos).astype(int), len(values) - 1)
    hi = np.minimum(lo + 1, len(values) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    out = values[lo] * (1 - w) + values[hi] * w  # NaN if either end is NaN (w == 0 keeps lo only)
    exact = (w == 0)
    out = np.where(exact, values[lo], out)
    return out


def load_match(match_id: str = DEFAULT_MATCH, data_dir: Path = None):
    """-> (Tracking at 25 fps, gt_codes, info). Codes follow possession_metrics (H0 A1 D2 X3)."""
    data_dir = Path(data_dir) if data_dir else default_dir(match_id)
    meta = json.loads((data_dir / "{}_match.json".format(match_id)).read_text(encoding="utf-8"))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    team_of_player = {p["id"]: (0 if p["team_id"] == home_id else 1)
                      for p in meta["players"] if p.get("team_id") in (home_id, away_id)}
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    centre = np.array(pitch) / 2.0

    frames, period, ball, rows = [], [], [], []
    with open(data_dir / "{}_tracking_extrapolated.jsonl".format(match_id), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            frames.append(o["frame"])
            period.append(o["period"] if o["period"] is not None else 0)
            b = o["ball_data"] or {}
            ok = bool(b.get("is_detected")) and b.get("x") is not None and b.get("y") is not None
            ball.append((b["x"], b["y"]) if ok else (np.nan, np.nan))
            rows.append({p["player_id"]: (p["x"], p["y"]) for p in o["player_data"]
                         if p.get("x") is not None and p["player_id"] in team_of_player})
    n_src = len(frames)
    if frames != list(range(n_src)):
        raise ValueError("tracking frames are not contiguous from 0")

    pids = sorted({pid for r in rows for pid in r})
    index = {p: i for i, p in enumerate(pids)}
    players = np.full((n_src, len(pids), 2), np.nan)
    for t, r in enumerate(rows):
        for pid, xy in r.items():
            players[t, index[pid]] = xy
    ball = np.array(ball, dtype=float)
    period = np.array(period)

    n = int(round(n_src * FPS / SRC_FPS))
    players25 = _interp(players, n) + centre
    ball25 = _interp(ball, n) + centre
    period25 = period[np.minimum((np.arange(n) * SRC_FPS / FPS).round().astype(int), n_src - 1)]

    # --- labels at 25 fps from phases of play (frames in the file are source-rate frames)
    phases = pd.read_csv(data_dir / "{}_phases_of_play.csv".format(match_id))
    gt = np.full(n, 3, dtype=np.int8)
    in_play = period25 > 0
    gt[in_play] = 2  # D unless a phase covers it
    for _, ph in phases.iterrows():
        a = int(round(ph["frame_start"] * FPS / SRC_FPS))
        b = int(round(ph["frame_end"] * FPS / SRC_FPS))
        team = 0 if ph["team_in_possession_id"] == home_id else 1
        seg = slice(a, min(b, n))
        covered = in_play[seg]
        gt[seg][covered] = team
    halves = []
    for p in sorted(set(period25[period25 > 0].tolist())):
        idx = np.flatnonzero(period25 == p)
        halves.append((int(idx[0]), int(idx[-1]) + 1))

    slot_team = np.array([team_of_player[p] for p in pids], dtype=np.int8)
    info = {
        "n_frames": n,
        "n_slots": len(pids),
        "ball_missing_rate": float(np.isnan(ball25[:, 0]).mean()),
        "ball_missing_rate_in_play": float(np.isnan(ball25[in_play, 0]).mean()),
        "halves": halves,
    }
    return Tracking(ball=ball25, players=players25, slot_team=slot_team), gt, info
