"""Shared types and constants for the external validation bench.

Cross-platform: pathlib only, no hard-coded paths. No competition data is read here.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Competition geometry/frame rate, published in DATA.md and games.py (see PROVENANCE.md).
PITCH_M = (105.3, 68.0)
FPS = 25

# Internal team codes (same convention as starter/baseline.py): 0 home, 1 away, -1 none.
HOME, AWAY, NONE = 0, 1, -1

# Label codes follow evaluation/possession_metrics.py: LABELS = ("H", "A", "D", "X").
H, A, D, X = 0, 1, 2, 3


@dataclass
class Tracking:
    """Tracking of one match in metres, frames x slots.

    ball:     (n, 2) float, NaN where the ball is not visible.
    players:  (n, P, 2) float, NaN where a slot is empty.
    slot_team:(P,) int8, 0 home / 1 away (known from metadata, like the competition ATD).
    """

    ball: np.ndarray
    players: np.ndarray
    slot_team: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])


def load_possession_metrics():
    """Import evaluation/possession_metrics.py by path (evaluation/ is not a package)."""
    path = REPO_ROOT / "evaluation" / "possession_metrics.py"
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
