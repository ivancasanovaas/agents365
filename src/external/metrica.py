"""Adapter for Metrica Sports Sample Game 3 (EPTS FIFA tracking + JSON events), an OPEN game that is
NOT the competition match. No kloppy needed: stdlib + numpy only.

Frame convention: tracking line i (0-based) is frame id i+1; event frames use the same ids.
Home team = the metadata's local team (FIFATMA).

Possession rule (algorithmic, see PROVENANCE.md and CHALLENGE.md's "control, not touch"):
- CONTROL events (PASS, CARRY, SHOT, RECOVERY, SET PIECE) set the possessing team from their start frame.
- CHALLENGE and BALL LOST never change the owner by themselves (no control gained).
- Dead ball: from a BALL OUT / FAULT RECEIVED / goal-SHOT event to the start of the next SET PIECE.
- Frames before the first event of a half are D.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .common import REPO_ROOT, Tracking
from .labels import labels_from_events

DEFAULT_DIR = REPO_ROOT / "ext_open" / "metrica_game3"
STEM = "Sample_Game_3"

CONTROL_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}
DEAD_START_TYPES = {"BALL OUT", "FAULT RECEIVED"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_metadata(path: Path):
    root = ET.parse(path).getroot()
    home_id = field_w = field_h = None
    player_team = {}
    channel_player = {}
    halves = {}
    for el in root.iter():
        name = _local(el.tag)
        if name == "Score":
            home_id = el.attrib["idLocalTeam"]
        elif name == "Player" and "teamId" in el.attrib:
            player_team[el.attrib["id"]] = el.attrib["teamId"]
        elif name == "PlayerChannel":
            channel_player[el.attrib["id"]] = el.attrib["playerId"]
        elif name == "FieldSize":
            field_w = float(el.find("Width").text)
            field_h = float(el.find("Height").text)
        elif name == "DataFormatSpecification":
            # Each line carries only the players on the pitch; the order can change per half.
            order = []
            for ref in el.iter():
                if _local(ref.tag) == "PlayerChannelRef" and ref.attrib["playerChannelId"].endswith("_x"):
                    order.append(channel_player[ref.attrib["playerChannelId"]])
            halves[len(halves) + 1] = (int(el.attrib["startFrame"]), int(el.attrib["endFrame"]), order)
    all_players = list(dict.fromkeys(channel_player.values()))  # stable, unique
    slot_team = np.array([0 if player_team[p] == home_id else 1 for p in all_players], dtype=np.int8)
    return slot_team, all_players, (field_w, field_h), halves


def read_period_runs(path: Path):
    """Half-open 0-based [start, stop) per period from the provider's first/second half bounds."""
    root = ET.parse(path).getroot()
    params = {}
    for el in root.iter():
        if _local(el.tag) == "ProviderParameter":
            name, value = el.find("Name"), el.find("Value")
            if name is not None and value is not None and value.text:
                params[name.text] = value.text
    return [
        (int(params["first_half_start"]) - 1, int(params["first_half_end"])),
        (int(params["second_half_start"]) - 1, int(params["second_half_end"])),
    ]


def read_tracking(path: Path, all_players, field: Tuple[float, float], halves):
    text = path.read_text()
    flat = np.array(text.replace(":", ",").replace(";", ",").replace("\n", ",").strip(",").split(","),
                    dtype=np.float64)  # "NaN" strings parse to nan
    n_on = len(next(iter(halves.values()))[2])
    per_line = 1 + 2 * n_on + 2
    if flat.size % per_line or any(len(h[2]) != n_on for h in halves.values()):
        raise ValueError("unexpected tracking layout: {} values per line expected".format(per_line))
    grid = flat.reshape(-1, per_line)
    frame_ids = grid[:, 0].astype(np.int64)
    scale = np.array(field)
    index = {p: i for i, p in enumerate(all_players)}
    players = np.full((len(grid), len(all_players), 2), np.nan)
    for start, end, order in halves.values():
        rows = np.flatnonzero((frame_ids >= start) & (frame_ids <= end))
        block = grid[rows, 1:1 + 2 * n_on].reshape(len(rows), n_on, 2) * scale
        for col, pid in enumerate(order):
            players[rows, index[pid]] = block[:, col]
    ball = grid[:, 1 + 2 * n_on:] * scale
    return frame_ids, ball, players


def _subtypes(event) -> List[str]:
    sub = event.get("subtypes")
    if sub is None:
        return []
    items = sub if isinstance(sub, list) else [sub]
    return [str(i.get("name", "")).upper() for i in items]


def derive_events(events, home_id: str, period_runs):
    """-> control_events [(idx, team)], dead_intervals [(start, stop)] in 0-based frame indices."""
    control, dead = [], []
    ordered = sorted(events, key=lambda e: (e["period"], e["start"]["frame"]))
    for k, ev in enumerate(ordered):
        kind = ev["type"]["name"].upper()
        start = ev["start"]["frame"] - 1
        team = 0 if ev["team"]["id"] == home_id else 1
        if kind in CONTROL_TYPES:
            control.append((start, team))
        is_goal = kind == "SHOT" and any("GOAL" in s for s in _subtypes(ev))
        if kind in DEAD_START_TYPES or is_goal:
            begin = (ev["end"]["frame"] if is_goal else ev["start"]["frame"]) - 1
            stop = None
            for nxt in ordered[k + 1:]:
                if nxt["type"]["name"].upper() == "SET PIECE" and nxt["period"] == ev["period"]:
                    stop = nxt["start"]["frame"] - 1
                    break
            if stop is None:
                stop = period_runs[ev["period"]][1]
            if stop > begin:
                dead.append((begin, stop))
    return sorted(control), dead


def load_game3(data_dir: Path = DEFAULT_DIR):
    """-> (Tracking, gt_codes, info). gt_codes follow possession_metrics (H0 A1 D2 X3)."""
    data_dir = Path(data_dir)
    slot_team, all_players, field, halves = read_metadata(data_dir / "{}_metadata.xml".format(STEM))
    frame_ids, ball, players = read_tracking(
        data_dir / "{}_tracking.txt".format(STEM), all_players, field, halves)
    n = len(frame_ids)
    if not np.array_equal(frame_ids, np.arange(1, n + 1)):
        raise ValueError("tracking frame ids are not contiguous from 1")
    runs = {h: (a - 1, b) for h, (a, b, _) in halves.items()}  # half-open, 0-based
    data = json.loads((data_dir / "{}_events.json".format(STEM)).read_text())["data"]
    root = ET.parse(data_dir / "{}_metadata.xml".format(STEM)).getroot()
    home_id = next(el.attrib["idLocalTeam"] for el in root.iter() if _local(el.tag) == "Score")
    control, dead = derive_events(data, home_id, runs)
    gt = labels_from_events(n, control, dead, runs.values())
    info = {
        "n_frames": n,
        "n_slots": len(slot_team),
        "ball_missing_rate": float(np.isnan(ball[:, 0]).mean()),
        "n_control_events": len(control),
        "n_dead_intervals": len(dead),
        "halves": read_period_runs(data_dir / "{}_metadata.xml".format(STEM)),
    }
    return Tracking(ball=ball, players=players, slot_team=slot_team), gt, info
