"""Save/load HMM parameters as small JSON files (derived from EXTERNAL games only, safe to commit)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hmm import HmmParams


def save_params(path, params: HmmParams, meta: dict) -> None:
    payload = {
        "meta": meta,
        "log_trans": params.log_trans.tolist(),
        "log_team": params.log_team.tolist(),
        "log_speed": params.log_speed.tolist(),
        "log_prior": params.log_prior.tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def load_params(path):
    payload = json.loads(Path(path).read_text())
    params = HmmParams(
        log_trans=np.array(payload["log_trans"]),
        log_team=np.array(payload["log_team"]),
        log_speed=np.array(payload["log_speed"]),
        log_prior=np.array(payload["log_prior"]),
    )
    return params, payload["meta"]
