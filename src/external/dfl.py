"""Adapter for the Sportec Open DFL Dataset (7 Bundesliga matches), an OPEN dataset of OTHER games, NOT the competition match.

Source: https://doi.org/10.6084/m9.figshare.28196177 (Bassek et al., Scientific Data 2025), CC BY 4.0; downloaded with
scripts/fetch_dfl.py into ext_open/dfl/. Files per match: DFL_02_01_matchinformation_*_<match>.xml and
DFL_04_03_positions_raw_observed_*_<match>.xml.

Tracking: one <FrameSet> per person and game section ("firstHalf", "secondHalf"), 25 Hz, pitch-centred metres
(x in [-52.5, 52.5], y in [-34, 34]); the BALL frame set also carries, per frame, `BallStatus` (1 alive, 0 dead) and
`BallPossession` (1 or 2). These are NATIVE per-frame labels with the same meaning as the competition's ground truth
("D where the ball is dead, otherwise the team the tracking names as owning it"), so no event-to-label rule is needed:
    label = D if BallStatus == 0 else (H if the possession code maps to the home team else A).
Which code maps to the home team is resolved per match by agreement with the nearest player to the ball on alive frames
(the DFL convention is 1 = home); it is checked, not assumed.

Output follows the other adapters: (Tracking at 25 fps, labels H0 A1 D2 X3, info with 'halves').
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .common import REPO_ROOT, Tracking

PITCH_DFL = (105.0, 68.0)
DEFAULT_DIR = REPO_ROOT / "ext_open" / "dfl"
SECTIONS = ("firstHalf", "secondHalf")


def _files(match_id: str, data_dir: Path) -> Tuple[Path, Path]:
    info = sorted(data_dir.glob("DFL_02_01_matchinformation_*_{}.xml".format(match_id)))
    pos = sorted(data_dir.glob("DFL_04_03_positions_raw_observed_*_{}.xml".format(match_id)))
    if not info or not pos:
        raise FileNotFoundError("DFL files for {} not found under {} (python scripts/fetch_dfl.py)".format(match_id, data_dir))
    return info[0], pos[0]


def match_ids(data_dir: Path = DEFAULT_DIR) -> List[str]:
    """Match ids with a positions file present (size > 1 MB)."""
    out = []
    for p in sorted(Path(data_dir).glob("DFL_04_03_positions_raw_observed_*_DFL-MAT-*.xml")):
        if p.stat().st_size > 1_000_000:
            out.append(p.stem.rsplit("_", 1)[-1])
    return out


def read_teams(path: Path) -> Dict[str, str]:
    """-> {'home': team id, 'guest': team id} from the match information."""
    roles = {}
    for el in ET.parse(path).getroot().iter("Team"):
        roles[el.get("Role")] = el.get("TeamId")
    if set(roles) != {"home", "guest"}:
        raise ValueError("expected a home and a guest team in {}".format(path))
    return roles


def _parse_positions(path: Path):
    """-> {section: {'ball': (N, x, y, poss, status), 'players': {person: (team, N, x, y)}}} (streaming)."""
    sections: Dict[str, dict] = {}
    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "FrameSet":
            continue
        sec, team, person = elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId")
        n, x, y, poss, status = [], [], [], [], []
        is_ball = team == "BALL"
        for f in elem:
            n.append(int(f.get("N")))
            x.append(float(f.get("X")))
            y.append(float(f.get("Y")))
            if is_ball:
                poss.append(int(f.get("BallPossession", 0)))
                status.append(int(f.get("BallStatus", 0)))
        entry = sections.setdefault(sec, {"ball": None, "players": {}})
        if is_ball:
            entry["ball"] = (np.array(n), np.array(x), np.array(y), np.array(poss), np.array(status))
        elif team != "referee":
            entry["players"][person] = (team, np.array(n), np.array(x), np.array(y))
        elem.clear()
    return sections


def load_match(match_id: str, data_dir: Path = None):
    """-> (Tracking, gt_codes, info) like the other external adapters (H0 A1 D2 X3; halves as half-open runs)."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DIR
    info_path, pos_path = _files(match_id, data_dir)
    roles = read_teams(info_path)
    sections = _parse_positions(pos_path)
    secs = [s for s in SECTIONS if s in sections and sections[s]["ball"] is not None]
    if not secs:
        raise ValueError("no ball frames in {}".format(pos_path))

    persons: Dict[str, int] = {}
    for s in secs:
        for pid in sections[s]["players"]:
            persons.setdefault(pid, len(persons))
    team_of = {}
    for s in secs:
        for pid, (team, *_rest) in sections[s]["players"].items():
            team_of[pid] = 0 if team == roles["home"] else 1
    slot_team = np.zeros(len(persons), dtype=np.int8)
    for pid, k in persons.items():
        slot_team[k] = team_of[pid]

    lengths = [len(sections[s]["ball"][0]) for s in secs]
    n = int(sum(lengths))
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(persons), 2), np.nan)
    poss = np.zeros(n, dtype=np.int64)
    status = np.zeros(n, dtype=np.int64)
    off = 0
    sec_bounds = []
    cx, cy = PITCH_DFL[0] / 2.0, PITCH_DFL[1] / 2.0
    for s, ln in zip(secs, lengths):
        bn, bx, by, bp, bs = sections[s]["ball"]
        if not np.array_equal(bn, np.arange(bn[0], bn[0] + ln)):
            raise ValueError("ball frames of {} are not contiguous".format(s))
        ball[off:off + ln, 0], ball[off:off + ln, 1] = bx + cx, by + cy
        poss[off:off + ln], status[off:off + ln] = bp, bs
        for pid, (team, pn, px, py) in sections[s]["players"].items():
            idx = pn - bn[0]
            ok = (idx >= 0) & (idx < ln)
            players[off + idx[ok], persons[pid], 0] = px[ok] + cx
            players[off + idx[ok], persons[pid], 1] = py[ok] + cy
        sec_bounds.append((off, off + ln))
        off += ln

    # graded runs: from the first to the last ALIVE frame of each section (pre-kickoff / post-whistle frames are not graded)
    halves = []
    for lo, hi in sec_bounds:
        alive = np.flatnonzero(status[lo:hi] == 1)
        if alive.size:
            halves.append((lo + int(alive[0]), lo + int(alive[-1]) + 1))

    tr = Tracking(ball=ball, players=players, slot_team=slot_team)
    home_code = _home_possession_code(tr, poss, status)
    gt = np.full(n, 3, dtype=np.int8)
    for lo, hi in halves:
        seg_poss, seg_status = poss[lo:hi], status[lo:hi]
        lab = np.where(seg_poss == home_code, 0, 1).astype(np.int8)
        lab[seg_status == 0] = 2
        gt[lo:hi] = lab
    info = {"n_frames": n, "n_slots": int(len(persons)), "ball_missing_rate": float(np.isnan(ball[:, 0]).mean()),
            "halves": halves, "home_possession_code": int(home_code)}
    return tr, gt, info


def _home_possession_code(tr: Tracking, poss: np.ndarray, status: np.ndarray) -> int:
    """Which BallPossession code (1 or 2) means the home team: the one whose agreement with the team of the nearest player to
    the ball on alive frames is higher. Raises if neither explains the data (agreement < 0.6)."""
    from ..model.frame_clf import build_features  # noqa: F401  (import cycle guard: only to fail early if the package is broken)
    diff = tr.players - tr.ball[:, None, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    dist[np.isnan(dist)] = np.inf
    nearest = dist.argmin(axis=1)
    team = tr.slot_team[nearest]
    ok = (status == 1) & np.isfinite(dist.min(axis=1)) & ((poss == 1) | (poss == 2))
    agree_1 = float(((team[ok] == 0) == (poss[ok] == 1)).mean())          # hypothesis: 1 = home
    best = max(agree_1, 1.0 - agree_1)
    if best < 0.6:
        raise ValueError("possession codes do not agree with the nearest-player team (agreement {:.2f})".format(best))
    return 1 if agree_1 >= 0.5 else 2
