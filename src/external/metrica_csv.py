"""Adapter for Metrica Sports Sample Games 1 and 2 (open CSV format), NOT the competition match.

Files per game in ext_open/metrica_game<N>/: Sample_Game_<N>_RawTrackingData_Home_Team.csv, ..._Away_Team.csv and
Sample_Game_<N>_RawEventsData.csv. Tracking rows: Period, Frame (1-based), Time [s], then x,y per player (normalised 0-1,
NaN when absent), the Home file ends with the ball. 25 fps, pitch 105 x 68 m. Events follow the same type vocabulary as
Sample Game 3, so labels reuse `metrica.derive_events` (control, not touch; dead ball from BALL OUT / FAULT RECEIVED /
goal to the next SET PIECE).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .common import REPO_ROOT, Tracking
from .labels import labels_from_events
from .metrica import derive_events

PITCH_METRICA = (105.0, 68.0)


def default_dir(game: int) -> Path:
    return REPO_ROOT / "ext_open" / "metrica_game{}".format(game)


def _read_tracking(path: Path):
    """-> (period (n,), frame (n,), xy (n, k, 2) normalised, has_ball). Header is three rows."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    body = rows[3:]
    period = np.array([int(r[0]) for r in body])
    frame = np.array([int(r[1]) for r in body])
    values = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r[3:]] for r in body])
    if values.shape[1] % 2:
        values = values[:, :-1]
    return period, frame, values.reshape(len(body), -1, 2)


def load_game(game: int, data_dir: Path = None):
    """-> (Tracking, gt_codes, info) like the other external adapters (H0 A1 D2 X3, halves as half-open runs)."""
    data_dir = Path(data_dir) if data_dir else default_dir(game)
    stem = "Sample_Game_{}".format(game)
    period, frame, home = _read_tracking(data_dir / "{}_RawTrackingData_Home_Team.csv".format(stem))
    period_a, frame_a, away = _read_tracking(data_dir / "{}_RawTrackingData_Away_Team.csv".format(stem))
    n = len(frame)
    if not np.array_equal(frame, np.arange(1, n + 1)) or not np.array_equal(frame, frame_a):
        raise ValueError("tracking frames are not contiguous from 1 or home/away files disagree")
    scale = np.array(PITCH_METRICA)
    ball = home[:, -1, :] * scale                       # the Home file ends with the ball pair
    home_p, away_p = home[:, :-1, :] * scale, away[:, :-1, :] * scale
    players = np.concatenate([home_p, away_p], axis=1)
    slot_team = np.array([0] * home_p.shape[1] + [1] * away_p.shape[1], dtype=np.int8)

    halves, runs = [], {}
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        halves.append((int(idx[0]), int(idx[-1]) + 1))
        runs[p] = (int(idx[0]), int(idx[-1]) + 1)

    events: List[Dict] = []
    with open(data_dir / "{}_RawEventsData.csv".format(stem), newline="") as handle:
        for r in csv.DictReader(handle):
            sub = (r["Subtype"] or "").strip()
            events.append({
                "type": {"name": r["Type"]},
                "subtypes": [{"name": sub}] if sub else None,
                "period": int(r["Period"]),
                "start": {"frame": int(r["Start Frame"])},
                "end": {"frame": int(r["End Frame"])},
                "team": {"id": r["Team"]},
            })
    control, dead = derive_events(events, "Home", runs)
    gt = labels_from_events(n, control, dead, runs.values())
    info = {
        "n_frames": n,
        "n_slots": int(players.shape[1]),
        "ball_missing_rate": float(np.isnan(ball[:, 0]).mean()),
        "n_control_events": len(control),
        "n_dead_intervals": len(dead),
        "halves": halves,
    }
    return Tracking(ball=ball, players=players, slot_team=slot_team), gt, info
