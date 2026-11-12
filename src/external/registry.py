"""Registry of the external (open, other-game) datasets and one loader signature for all of them.

    from src.external.registry import available, load
    names = available()              # only those whose files exist under ext_open/
    clean, gt, info = load(name)     # (Tracking, labels H0 A1 D2 X3, info with 'halves')

All of them are OTHER games with algorithmically derived labels (see PROVENANCE.md); none is the competition match.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .common import REPO_ROOT

SKILLCORNER_IDS = ("1874553", "1886347", "1899585", "1925299")
METRICA_CSV_GAMES = (1, 2)
# UNTOUCHED HOLD-OUT (EXP-014): never listed by `available()`, never used for training, tuning or selection. Only
# `load_holdout` can read them and only src/external/run_exp014.py calls it.
HOLDOUT_IDS = ("1927964", "1953632", "1959846", "1986691")


def _loaders() -> Dict[str, Callable]:
    from . import metrica, metrica_csv, skillcorner
    out: Dict[str, Callable] = {"metrica_game3": metrica.load_game3}
    for g in METRICA_CSV_GAMES:
        out["metrica_game{}".format(g)] = (lambda g=g: metrica_csv.load_game(g))
    for m in SKILLCORNER_IDS:
        out["skillcorner_{}".format(m)] = (lambda m=m: skillcorner.load_match(m))
    return out


def _marker(name: str):
    root = REPO_ROOT / "ext_open" / name
    if name == "metrica_game3":
        return root / "Sample_Game_3_tracking.txt"
    if name.startswith("metrica_game"):
        g = name[len("metrica_game"):]
        return root / "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g)
    m = name[len("skillcorner_"):]
    return root / "{}_tracking_extrapolated.jsonl".format(m)


def available() -> List[str]:
    """Names of datasets whose tracking file is present (and not a Git LFS pointer)."""
    names = []
    for name in _loaders():
        path = _marker(name)
        if path.exists() and path.stat().st_size > 1_000_000:
            names.append(name)
    return names


def holdout_available() -> List[str]:
    """Hold-out matches present on disk. Used ONLY by EXP-014."""
    out = []
    for m in HOLDOUT_IDS:
        path = REPO_ROOT / "ext_open" / "skillcorner_{}".format(m) / "{}_tracking_extrapolated.jsonl".format(m)
        if path.exists() and path.stat().st_size > 1_000_000:
            out.append("skillcorner_{}".format(m))
    return out


def load_holdout(name: str):
    if not name.startswith("skillcorner_") or name[len("skillcorner_"):] not in HOLDOUT_IDS:
        raise KeyError("{!r} is not a hold-out match".format(name))
    from . import skillcorner
    return skillcorner.load_match(name[len("skillcorner_"):])


def load(name: str):
    loaders = _loaders()
    if name not in loaders:
        raise KeyError("unknown external dataset {!r}; known: {}".format(name, sorted(loaders)))
    return loaders[name]()
