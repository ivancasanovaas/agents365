"""Reading the competition tracking (ATD) and the graded-window constants.

This module only ever runs on the machine that holds the match data (the DGX Spark). It must never
print rows of data: callers print aggregates only.

Raw ATD line (documented in starter/baseline.py): "<frame_id>:<slot>;<slot>;...:<ball>", each group
"x,y,speed" with "NaN" for a missing observation; coordinates are normalised 0-1; frame_id counts
from 1 while submission rows count from 0.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .external.common import REPO_ROOT, Tracking

for _p in (str(REPO_ROOT), str(REPO_ROOT / "starter")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIELD_SEP, GROUP_SEP = ":", ";"


def read_atd(raw_path, n_frames: int, pitch: Tuple[float, float], n_slots: int):
    """-> (ball (n,2) m, players (n,n_slots,2) m), NaN where unobserved. Streams the file."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            b = parts[2]
            if b and b[0] != "N":
                bx, by = b.split(",")[:2]
                ball[row] = (float(bx) * length, float(by) * width)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots:
                    raise ValueError("more track slots than team metadata entries")
                if group and group[0] != "N":
                    x, y = group.split(",")[:2]
                    players[row, slot] = (float(x) * length, float(y) * width)
    return ball, players


def load_competition(data_dir=None, game: str = None):
    """-> (Tracking, graded_runs [(start, stop) half-open], forced_dead [(start, stop)])."""
    import games  # organiser module at the repo root
    from baseline import team_of_each_track  # starter/baseline.py
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # published kickoff frames

    game = game or games.GAME
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, "tactical")
    cfg = games.GAMES[game]
    slot_team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, cfg["n_video_frames"], cfg["pitch"], len(slot_team))
    runs = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    # FAQ.md: the second half is graded from the first half-block frame but kickoff is later;
    # the frames in between are graded and are all D (published rule, not a hand observation).
    forced_dead = [(runs[1][0], KICKOFF_VIDEO_FRAMES["2nd Half"])]
    return Tracking(ball=ball, players=players, slot_team=slot_team), runs, forced_dead
