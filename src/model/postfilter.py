"""Post-processing of a decoded label sequence: minimum possession duration.

Deterministic, numpy only. Labels follow possession_metrics: H=0, A=1, D=2, X=3. Works inside the graded runs only.
The duration threshold is never hand-set: it comes from `spell_duration_quantile` on EXTERNAL labelled games.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _rle(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run starts, lengths and labels of a 1-D label array."""
    if seg.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [seg.size]))
    return starts, ends - starts, seg[starts]


def spell_lengths(codes: np.ndarray, runs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Lengths (frames) of maximal H or A runs inside the given runs (a possession, as the scorer defines it)."""
    out: List[np.ndarray] = []
    for lo, hi in runs:
        _, lens, labs = _rle(codes[lo:hi])
        out.append(lens[labs <= 1])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def spell_duration_quantile(gt: np.ndarray, runs: Sequence[Tuple[int, int]], q: float) -> int:
    """q-quantile (frames, rounded) of the true possession durations of a labelled game."""
    lens = spell_lengths(gt, runs)
    if lens.size == 0:
        raise ValueError("no possessions in the labelled runs")
    return int(round(float(np.quantile(lens, q))))


def min_duration_filter(codes: np.ndarray, min_frames: int, runs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Remove H/A runs shorter than `min_frames`. Repeatedly takes the shortest offending run (leftmost on ties) and
    absorbs it into the LONGER neighbouring H/A run, or relabels it D when neither neighbour is H/A, until none is left.
    Frames outside the runs are untouched; the input array is not modified."""
    out = codes.copy()
    for lo, hi in runs:
        seg = out[lo:hi]
        while True:
            starts, lens, labs = _rle(seg)
            short = np.flatnonzero((labs <= 1) & (lens < min_frames))
            if short.size == 0:
                break
            i = int(short[np.argmin(lens[short])])
            cands = []
            if i > 0 and labs[i - 1] <= 1:
                cands.append((int(lens[i - 1]), int(labs[i - 1])))
            if i + 1 < len(starts) and labs[i + 1] <= 1:
                cands.append((int(lens[i + 1]), int(labs[i + 1])))
            new = max(cands)[1] if cands else 2
            seg[starts[i]:starts[i] + lens[i]] = new
        out[lo:hi] = seg
    return out
