"""Save/load HMM parameters as small JSON files (derived from EXTERNAL games only, safe to commit)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .frame_clf import FEATURE_NAMES, MlpModel, SoftmaxModel
from .hmm import HmmParams


def save_params(path, params: HmmParams, meta: dict) -> None:
    payload = {
        "meta": meta,
        "log_trans": params.log_trans.tolist(),
        "log_team": params.log_team.tolist(),
        "log_speed": params.log_speed.tolist(),
        "log_prior": params.log_prior.tolist(),
        "log_extra": {k: v.tolist() for k, v in params.log_extra.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def save_softmax(path, model, meta: dict) -> None:
    """Softmax emission model (EXP-010): weights, standardisation and class prior as small JSON."""
    payload = {"meta": meta, "feature_names": list(FEATURE_NAMES), "W": model.W.tolist(), "mean": model.mean.tolist(),
               "std": model.std.tolist(), "log_prior": model.log_prior.tolist()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def load_softmax(path):
    payload = json.loads(Path(path).read_text())
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("softmax params were fitted with a different feature set; refit with python -m src.fit_params")
    model = SoftmaxModel(np.array(payload["W"]), np.array(payload["mean"]), np.array(payload["std"]),
                         np.array(payload["log_prior"]))
    return model, payload["meta"]


def save_mlp(path, model, meta: dict) -> None:
    """One-hidden-layer MLP emission model (EXP-013/014) as small JSON."""
    payload = {"meta": meta, "feature_names": list(FEATURE_NAMES), "W1": model.W1.tolist(), "b1": model.b1.tolist(),
               "W2": model.W2.tolist(), "b2": model.b2.tolist(), "mean": model.mean.tolist(), "std": model.std.tolist(),
               "log_prior": model.log_prior.tolist()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def load_mlp(path):
    payload = json.loads(Path(path).read_text())
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("MLP params were fitted with a different feature set; refit with python -m src.fit_params")
    model = MlpModel(*(np.array(payload[k]) for k in ("W1", "b1", "W2", "b2", "mean", "std", "log_prior")))
    return model, payload["meta"]


def load_params(path):
    payload = json.loads(Path(path).read_text())
    params = HmmParams(
        log_trans=np.array(payload["log_trans"]),
        log_team=np.array(payload["log_team"]),
        log_speed=np.array(payload["log_speed"]),
        log_prior=np.array(payload["log_prior"]),
        log_extra={k: np.array(v) for k, v in payload.get("log_extra", {}).items()},
    )
    return params, payload["meta"]
