"""Aggregate-only profile of the competition inputs (tracking, Smart Tagging, optionally our output).

    python -m src.profile_datasets [--data-dir PATH] [--submission outputs/submission.csv] [--out outputs/profile.md]

Run it on the machine that holds the match data. It writes ONE markdown file made only of aggregates
(counts, shares, quantiles). It never prints or stores a row, a frame index list or a position.
The profile is the only thing that may be handed to the local LLM (src/local_analyst.py) or pasted back
to a human/agent that must not see the data.

Every number below that looks like a threshold (e.g. 0.5 m/s, 1 s, 2 m) is a REPORTING LEVEL used to
describe the data. None of them is a model constant and none feeds the pipeline. If a level is later
promoted to a feature threshold it needs its own row in PROVENANCE.md.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .external.common import FPS, PITCH_M, REPO_ROOT, Tracking

QS = (0.10, 0.50, 0.90, 0.99)
SPEED_LEVELS_MS = (10.0, 20.0, 30.0, 40.0)   # reporting levels for the speed tail (not thresholds)
STILL_BALL_MS = 0.5                          # reporting level: "ball hardly moves"
STILL_CROWD_MS = 0.5                         # reporting level: "median player hardly moves"
MIN_STILL_S = 1.0                            # reporting level: shortest stillness run counted
EDGE_NEAR_M = 2.0                            # reporting level: "close to a line"


# ----------------------------------------------------------------------------------------- helpers
def _run_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start index and length of every maximal True run of a 1-D bool array."""
    if mask.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return starts, ends - starts


def _runs_in_graded(mask: np.ndarray, runs: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Run starts (absolute frame index) and lengths, never crossing a graded-half boundary."""
    starts, lengths = [], []
    for lo, hi in runs:
        s, ln = _run_bounds(mask[lo:hi])
        starts.append(s + lo)
        lengths.append(ln)
    if not starts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(starts), np.concatenate(lengths)


def _graded_mask(n: int, runs: Sequence[Tuple[int, int]]) -> np.ndarray:
    g = np.zeros(n, dtype=bool)
    for lo, hi in runs:
        g[lo:hi] = True
    return g


def _quant(values: np.ndarray, unit: str = "", digits: int = 2) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n=0"
    qs = np.quantile(values, QS)
    body = " ".join("p{}={:.{d}f}".format(int(q * 100), v, d=digits) for q, v in zip(QS, qs))
    return "n={} {} max={:.{d}f}{}".format(values.size, body, values.max(), (" " + unit) if unit else "", d=digits)


def _pct(num: float, den: float) -> str:
    return "n/a" if den == 0 else "{:.2f}%".format(100.0 * num / den)


def _dist_to_edge(xy: np.ndarray) -> np.ndarray:
    """Signed distance (m) to the nearest touch/goal line; negative = outside the pitch."""
    length, width = PITCH_M
    return np.minimum.reduce([xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]])


# ----------------------------------------------------------------------------------------- sections
Section = Tuple[str, List[str]]


def structure_section(tr: Tracking, runs, g) -> Section:
    n = tr.n_frames
    lines = [
        "frames={} graded={} ungraded={} fps={}".format(n, int(g.sum()), int((~g).sum()), FPS),
        "graded runs (count)={}".format(len(runs)),
        "ball array {} {} | players array {} {}".format(tr.ball.shape, tr.ball.dtype, tr.players.shape, tr.players.dtype),
        "track slots={} (home {} / away {})".format(
            tr.players.shape[1], int((tr.slot_team == 0).sum()), int((tr.slot_team == 1).sum())),
    ]
    return "Structure", lines


def nulls_and_ranges_section(tr: Tracking, g) -> Section:
    visible = np.isfinite(tr.ball[:, 0])
    seen = np.isfinite(tr.players[:, :, 0])
    per_frame = seen.sum(axis=1)
    lines = [
        "ball visible: all {} | graded {}".format(_pct(visible.sum(), len(visible)), _pct((visible & g).sum(), g.sum())),
        "players seen per graded frame: {}".format(_quant(per_frame[g].astype(float), digits=1)),
        "graded frames with more than 22 players seen: {}".format(_pct((per_frame[g] > 22).sum(), g.sum())),
        "slot occupancy (share of graded frames seen), quantiles over slots: {}".format(
            _quant(seen[g].mean(axis=0), digits=3)),
    ]
    if visible.any():
        b = tr.ball[visible]
        lines.append("ball x range m [{:.1f}, {:.1f}] | y range m [{:.1f}, {:.1f}]".format(
            b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()))
    anyp = tr.players[np.isfinite(tr.players[:, :, 0])]
    if anyp.size:
        lines.append("players x range m [{:.1f}, {:.1f}] | y range m [{:.1f}, {:.1f}]".format(
            anyp[:, 0].min(), anyp[:, 0].max(), anyp[:, 1].min(), anyp[:, 1].max()))
    return "Nulls and ranges", lines


def ball_gap_section(tr: Tracking, runs, g) -> Section:
    missing = ~np.isfinite(tr.ball[:, 0])
    starts, lengths = _runs_in_graded(missing, runs)
    seconds = lengths / FPS
    total_missing = int(lengths.sum())
    lines = ["missing-ball runs inside graded halves: count={}".format(len(lengths)),
             "run length (s): {}".format(_quant(seconds))]
    for lo, hi, name in ((0, 1.0, "<= 1 s"), (1.0, 2.0, "1-2 s"), (2.0, 5.0, "2-5 s"), (5.0, np.inf, "> 5 s")):
        sel = (seconds > lo) & (seconds <= hi) if lo > 0 else (seconds <= hi)
        lines.append("share of missing frames in runs {}: {} ({} runs)".format(
            name, _pct(lengths[sel].sum(), total_missing), int(sel.sum())))
    return "Ball gaps (graded)", lines


def ball_speed_section(tr: Tracking, g) -> Section:
    step = np.linalg.norm(np.diff(tr.ball, axis=0), axis=1) * FPS      # NaN if either frame missing
    ok = g[:-1] & g[1:] & np.isfinite(step)
    s = step[ok]
    lines = ["ball step speed (m/s) between consecutive visible graded frames: {}".format(_quant(s))]
    for level in SPEED_LEVELS_MS:
        lines.append("share of steps above {:.0f} m/s: {} ({} steps)".format(
            level, _pct((s > level).sum(), s.size), int((s > level).sum())))
    return "Ball speed", lines


def out_of_bounds_section(tr: Tracking, runs, g) -> Section:
    visible = np.isfinite(tr.ball[:, 0])
    x, y = tr.ball[:, 0], tr.ball[:, 1]
    length, width = PITCH_M
    outside = visible & ((x < 0) | (x > length) | (y < 0) | (y > width))
    _, lens = _runs_in_graded(outside, runs)
    lines = [
        "visible graded frames with ball outside the pitch rectangle: {}".format(
            _pct((outside & g).sum(), (visible & g).sum())),
        "outside runs: count={} length (s) {}".format(len(lens), _quant(lens / FPS)),
    ]
    missing = ~visible
    starts, _ = _runs_in_graded(missing, runs)
    starts = starts[starts > 0]
    last = tr.ball[starts - 1]
    last = last[np.isfinite(last[:, 0])]
    if last.size:
        d = _dist_to_edge(last)
        lines.append("distance to nearest line of the LAST visible ball before a gap (m): {}".format(_quant(d)))
        lines.append("gaps whose last visible ball is within {:.0f} m of a line or outside: {} ({} of {})".format(
            EDGE_NEAR_M, _pct((d <= EDGE_NEAR_M).sum(), d.size), int((d <= EDGE_NEAR_M).sum()), d.size))
    return "Ball out of bounds", lines


def stillness_section(tr: Tracking, runs, g) -> Section:
    min_len = int(round(MIN_STILL_S * FPS))
    step = np.linalg.norm(np.diff(tr.ball, axis=0), axis=1) * FPS
    ball_still = np.concatenate((np.isfinite(step) & (step < STILL_BALL_MS), [False]))
    _, bl = _runs_in_graded(ball_still, runs)
    bl = bl[bl >= min_len]
    disp = np.linalg.norm(np.diff(tr.players, axis=0), axis=2) * FPS      # (n-1, P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        crowd = np.nanmedian(disp, axis=1)
    crowd = np.concatenate((crowd, [np.nan]))
    crowd_still = np.isfinite(crowd) & (crowd < STILL_CROWD_MS)
    _, cl = _runs_in_graded(crowd_still, runs)
    cl = cl[cl >= min_len]
    lines = [
        "ball stationary (< {:.1f} m/s) runs >= {:.0f} s: count={} length (s) {}".format(
            STILL_BALL_MS, MIN_STILL_S, len(bl), _quant(bl / FPS)),
        "median-player speed per graded frame (m/s): {}".format(_quant(crowd[g])),
        "median player hardly moving (< {:.1f} m/s) runs >= {:.0f} s: count={} length (s) {}".format(
            STILL_CROWD_MS, MIN_STILL_S, len(cl), _quant(cl / FPS)),
        "graded frames in such crowd-still runs: {}".format(_pct(int(cl.sum()), int(g.sum()))),
    ]
    return "Stillness", lines


def proximity_section(tr: Tracking, g) -> Section:
    diff = tr.players - tr.ball[:, None, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    dist[np.isnan(dist)] = np.inf
    idx = dist.argmin(axis=1)
    best = dist[np.arange(len(dist)), idx]
    ok = g & np.isfinite(best)
    d = best[ok]
    pos = tr.players[np.arange(len(dist)), idx]
    length, width = PITCH_M
    off = (pos[:, 0] < 0) | (pos[:, 0] > length) | (pos[:, 1] < 0) | (pos[:, 1] > width)
    lines = [
        "nearest-player distance (m), visible-ball graded frames: {}".format(_quant(d)),
        "share farther than 4 m: {} | farther than 8 m: {}".format(
            _pct((d > 4).sum(), d.size), _pct((d > 8).sum(), d.size)),
        "nearest tracked object is OUTSIDE the pitch rectangle: {} of visible-ball graded frames".format(
            _pct((off & ok).sum(), ok.sum())),
        "... and within 4 m of the ball: {}".format(_pct((off & ok & (best <= 4.0)).sum(), ok.sum())),
    ]
    return "Ball-player proximity", lines


def d_proxy_section(tr: Tracking, runs, g) -> Section:
    """Descriptive proxy for how much of the match could be dead ball. NOT a label, NOT a model input."""
    min_len = int(round(MIN_STILL_S * FPS))
    visible = np.isfinite(tr.ball[:, 0])
    x, y = tr.ball[:, 0], tr.ball[:, 1]
    length, width = PITCH_M
    outside = visible & ((x < 0) | (x > length) | (y < 0) | (y > width))
    step = np.linalg.norm(np.diff(tr.ball, axis=0), axis=1) * FPS
    still = np.concatenate((np.isfinite(step) & (step < STILL_BALL_MS), [False]))
    s_start, s_len = _runs_in_graded(still, runs)
    still_long = np.zeros(tr.n_frames, dtype=bool)
    for a, ln in zip(s_start, s_len):
        if ln >= min_len:
            still_long[a:a + ln] = True
    union = (outside | still_long) & g
    lines = [
        "graded frames with ball outside pitch: {}".format(_pct((outside & g).sum(), g.sum())),
        "graded frames inside a ball-stationary run >= {:.0f} s: {}".format(MIN_STILL_S, _pct((still_long & g).sum(), g.sum())),
        "union of the two (a coarse proxy, descriptive only): {}".format(_pct(union.sum(), g.sum())),
    ]
    return "Dead-ball proxy (descriptive)", lines


def tagging_section(events: Sequence[dict], n: int, g) -> Section:
    lines = ["instances={}".format(len(events))]
    by_code: Dict[str, List[float]] = {}
    for ev in events:
        by_code.setdefault(str(ev.get("code")), []).append((ev["end_frame"] - ev["start_frame"]) / FPS)
    for code in sorted(by_code, key=lambda c: -len(by_code[c])):
        lines.append("code {}: n={} duration (s) {}".format(code, len(by_code[code]), _quant(np.array(by_code[code]), digits=1)))
    teams: Dict[str, int] = {}
    for ev in events:
        t = str(ev.get("tags", {}).get("Team"))
        teams[t] = teams.get(t, 0) + 1
    lines.append("Team tag counts: {}".format(", ".join("{}={}".format(k, v) for k, v in sorted(teams.items()))))
    types: Dict[str, List[float]] = {}
    for ev in events:
        if str(ev.get("code")).strip().lower() == "set pieces":
            types.setdefault(str(ev.get("tags", {}).get("Type")), []).append((ev["end_frame"] - ev["start_frame"]) / FPS)
    for t in sorted(types):
        lines.append("set piece type {}: n={} duration (s) {}".format(t, len(types[t]), _quant(np.array(types[t]), digits=1)))
    union = np.zeros(n, dtype=bool)
    sp = np.zeros(n, dtype=bool)
    for ev in events:
        lo, hi = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        union[lo:hi] = True
        if str(ev.get("code")).strip().lower() == "set pieces":
            sp[lo:hi] = True
    lines.append("graded frames covered by any tagged interval: {}".format(_pct((union & g).sum(), g.sum())))
    lines.append("graded frames inside a SET PIECES interval: {}".format(_pct((sp & g).sum(), g.sum())))
    return "Smart Tagging", lines


def _crowd_speed(tr: Tracking) -> np.ndarray:
    """Median over visible slots of the per-frame speed (m/s); NaN where nothing is measurable."""
    disp = np.linalg.norm(np.diff(tr.players, axis=0), axis=2) * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(disp, axis=1)
    return np.concatenate((med, [np.nan]))


def _cue_starts(tr: Tracking, runs) -> Dict[str, np.ndarray]:
    """Start frames of three physical dead-ball cues. Used only to describe alignment."""
    length, width = PITCH_M
    visible = np.isfinite(tr.ball[:, 0])
    x, y = tr.ball[:, 0], tr.ball[:, 1]
    outside = visible & ((x < 0) | (x > length) | (y < 0) | (y > width))
    o_start, _ = _runs_in_graded(outside, runs)
    m_start, _ = _runs_in_graded(~visible, runs)
    m_start = m_start[m_start > 0]
    last = tr.ball[m_start - 1]
    ok = np.isfinite(last[:, 0])
    near = np.zeros(len(m_start), dtype=bool)
    near[ok] = _dist_to_edge(last[ok]) <= EDGE_NEAR_M
    crowd = _crowd_speed(tr)
    c_start, c_len = _runs_in_graded(np.isfinite(crowd) & (crowd < STILL_CROWD_MS), runs)
    c_start = c_start[c_len >= int(round(MIN_STILL_S * FPS))]
    return {"ball outside pitch": o_start, "gap after near-line ball": m_start[near], "crowd still": c_start}


def alignment_section(tr: Tracking, runs, events: Sequence[dict], g) -> Section:
    """How Smart Tagging set pieces line up with physical cues. Signed offsets in seconds, aggregates only."""
    sp = np.array(sorted(ev["start_frame"] for ev in events
                         if str(ev.get("code")).strip().lower() == "set pieces"
                         and "kick off" not in str(ev.get("tags", {}).get("Type")).lower()), dtype=np.int64)
    lines = ["set pieces considered (kick offs excluded): {}".format(len(sp))]
    if sp.size == 0:
        return "Set piece alignment", lines
    win = 10 * FPS
    for name, starts in _cue_starts(tr, runs).items():
        starts = np.sort(starts)
        if starts.size == 0:
            lines.append("cue {}: n=0".format(name))
            continue
        pos = np.searchsorted(starts, sp)
        left = starts[np.clip(pos - 1, 0, starts.size - 1)]
        right = starts[np.clip(pos, 0, starts.size - 1)]
        nearest = np.where(np.abs(left - sp) <= np.abs(right - sp), left, right)
        off = (nearest - sp) / FPS
        hit = np.abs(off) <= 10.0
        in_window = np.zeros(starts.size, dtype=bool)
        for s in sp:
            in_window |= (starts >= s - win) & (starts <= s + 35 * FPS)
        lines.append("cue {}: n={} | set pieces with a cue within +-10 s: {} ({} of {}) | signed offset cue-minus-tag (s) {}".format(
            name, starts.size, _pct(hit.sum(), sp.size), int(hit.sum()), sp.size, _quant(off[hit], digits=1)))
        lines.append("cue {}: share of cues falling in [-10 s, +35 s] of some set-piece start: {}".format(
            name, _pct(in_window.sum(), starts.size)))
    return "Set piece alignment", lines


def tag_consistency_section(events: Sequence[dict], codes: np.ndarray, g, home_key: str = "barcel") -> Section:
    """Agreement between OUR labels and the tagged Team. Internal consistency, NOT accuracy."""
    total = {}
    match = {}
    for ev in events:
        team = str(ev.get("tags", {}).get("Team")).strip().lower()
        if team == "n/a" or team == "none":
            continue
        side = 0 if team.startswith(home_key) else 1
        # In Smart Tagging, codes come in pairs over the same interval (e.g. PROGRESSION / DEFENDING IN MIDDLE
        # THIRD); the Team of the DEFENDING* and DEFENSIVE TRANSITION codes is the defending side, so the
        # possessing side is the other one.
        code_name = str(ev.get("code")).strip().upper()
        if code_name.startswith("DEFENDING") or code_name == "DEFENSIVE TRANSITION":
            side = 1 - side
        lo, hi = max(0, ev["start_frame"]), min(len(codes), ev["end_frame"])
        seg = codes[lo:hi]
        gg = g[lo:hi] & (seg <= 1)
        code = str(ev.get("code"))
        total[code] = total.get(code, 0) + int(gg.sum())
        match[code] = match.get(code, 0) + int(((seg == side) & gg).sum())
    lines = ["overall agreement with tagged Team (frames where we say H or A): {}".format(
        _pct(sum(match.values()), sum(total.values())))]
    for code in sorted(total, key=lambda c: -total[c])[:8]:
        lines.append("code {}: {} over {} frames".format(code, _pct(match[code], total[code]), total[code]))
    return "Our labels vs tagged Team (consistency only)", lines


def d_overlap_section(tr: Tracking, runs, events: Optional[Sequence[dict]], codes: np.ndarray, g) -> Section:
    is_d = codes == 2
    length, width = PITCH_M
    x, y = tr.ball[:, 0], tr.ball[:, 1]
    outside = np.isfinite(x) & ((x < 0) | (x > length) | (y < 0) | (y > width))
    crowd = _crowd_speed(tr)
    crowd_still = np.zeros(tr.n_frames, dtype=bool)
    c_start, c_len = _runs_in_graded(np.isfinite(crowd) & (crowd < STILL_CROWD_MS), runs)
    for a, ln in zip(c_start, c_len):
        if ln >= int(round(MIN_STILL_S * FPS)):
            crowd_still[a:a + ln] = True
    lines = [
        "share of frames with ball outside pitch that we label D: {}".format(_pct((is_d & outside & g).sum(), (outside & g).sum())),
        "share of crowd-still frames that we label D: {}".format(_pct((is_d & crowd_still & g).sum(), (crowd_still & g).sum())),
    ]
    if events is not None:
        sp = np.zeros(tr.n_frames, dtype=bool)
        for ev in events:
            if str(ev.get("code")).strip().lower() == "set pieces":
                sp[max(0, ev["start_frame"]):min(tr.n_frames, ev["end_frame"])] = True
        lines.append("share of set-piece-interval graded frames that we label D: {}".format(_pct((is_d & sp & g).sum(), (sp & g).sum())))
        lines.append("share of our graded D frames inside a set-piece interval: {}".format(_pct((is_d & sp & g).sum(), (is_d & g).sum())))
    return "Our D vs physical cues", lines


def feature_distribution_section(tr: Tracking, g) -> Section:
    """Share of graded frames per symbol of the EXP-003 streams, to compare against the degraded external bench."""
    from .model.hmm import EXTRA_NSYM, extract_extra
    extra = extract_extra(tr, ("out", "crowd"))
    lines = []
    for name, sym in extra.items():
        s = sym[g]
        valid = s >= 0
        shares = [float((s[valid] == k).mean()) if valid.any() else float("nan") for k in range(EXTRA_NSYM[name])]
        lines.append("stream {}: valid share of graded frames {} | symbol shares {}".format(
            name, _pct(valid.sum(), s.size), " ".join("{:.3f}".format(v) for v in shares)))
    return "EXP-003 stream distributions (graded)", lines


def submission_section(codes: np.ndarray, g) -> Section:
    """Aggregates of OUR OWN output (H=0 A=1 D=2 X=3). Counts and durations only."""
    lines = []
    for i, name in enumerate("HADX"):
        lines.append("label {}: {} of all frames | {} of graded frames".format(
            name, _pct((codes == i).sum(), codes.size), _pct(((codes == i) & g).sum(), g.sum())))
    for i, name in ((0, "H"), (1, "A"), (2, "D")):
        _, ln = _run_bounds(codes == i)
        lines.append("{} runs: count={} duration (s) {}".format(name, len(ln), _quant(ln / FPS, digits=1)))
    alive = np.where(codes <= 1, codes, -1)
    sw = ((alive[1:] != alive[:-1]) & (alive[1:] >= 0) & (alive[:-1] >= 0)).sum()
    lines.append("direct H<->A switches: {}".format(int(sw)))
    return "Our submission (aggregates)", lines


# ----------------------------------------------------------------------------------------- assemble
def build_profile(tr: Tracking, runs: Sequence[Tuple[int, int]], events: Optional[Sequence[dict]] = None,
                  codes: Optional[np.ndarray] = None) -> List[Section]:
    g = _graded_mask(tr.n_frames, runs)
    sections = [
        structure_section(tr, runs, g),
        nulls_and_ranges_section(tr, g),
        ball_gap_section(tr, runs, g),
        ball_speed_section(tr, g),
        out_of_bounds_section(tr, runs, g),
        stillness_section(tr, runs, g),
        proximity_section(tr, g),
        d_proxy_section(tr, runs, g),
        feature_distribution_section(tr, g),
    ]
    if events is not None:
        sections.append(tagging_section(events, tr.n_frames, g))
        sections.append(alignment_section(tr, runs, events, g))
    if codes is not None:
        codes = np.asarray(codes)
        sections.append(submission_section(codes, g))
        sections.append(d_overlap_section(tr, runs, events, codes, g))
        if events is not None:
            sections.append(tag_consistency_section(events, codes, g))
    return sections


def render(sections: Iterable[Section]) -> str:
    out = ["# Dataset profile (aggregates only)", "",
           "Reporting levels in this file are descriptive, not model constants.", ""]
    for title, lines in sections:
        out.append("## " + title)
        out.extend("- " + ln for ln in lines)
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    ap.add_argument("--submission", default=None, help="optional CSV of our own labels, summarised as aggregates")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "profile.md"))
    args = ap.parse_args(argv)

    from .competition import load_competition
    tr, runs, _ = load_competition(args.data_dir)

    events = None
    try:
        import games
        from load_smart_tagging import load as load_tags   # path set up by src.competition
        events = load_tags(games.GAME, args.data_dir)
    except Exception as exc:  # tagging is optional for the profile; say so, never print data
        print("smart tagging skipped: {}".format(type(exc).__name__))

    codes = None
    if args.submission:
        from .external.common import load_possession_metrics
        pm = load_possession_metrics()
        codes = pm.label_codes(pm.read_label_csv(args.submission))

    text = render(build_profile(tr, runs, events, codes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("wrote {} ({} lines, {} chars)".format(out, text.count("\n"), len(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
