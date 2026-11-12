"""Fit the HMM on EXTERNAL games and save the parameters (no competition data involved).

    python -m src.fit_params            # needs ext_open/metrica_game3 and ext_open/skillcorner_* (see PROVENANCE.md)

The parameters are estimated by counting on the POOLED external games (Metrica Sample Game 3 + SkillCorner 1874553),
each degraded with the profile-calibrated degrader. Pooling matters: some dead-ball cues (a ball outside the pitch for
more than 2 s) have no training frames in Metrica alone (EXP-004 support report). Pooling cannot be validated
leave-one-game-out with only two games; the single-game versions were validated by transfer (EXP-002/003/004).
Writes models/hmm_params.json.
"""
from __future__ import annotations

import argparse

import numpy as np

from .external.common import REPO_ROOT
from .external.degrade import calibrated_params, degrade
from .external.metrica import load_game3
from .external.skillcorner import load_match
from .model.hmm import concat_evidence, extract_evidence, fit
from .model.io import save_params

DEGRADE_SEED = 999
STREAMS = ("out", "crowd")     # variant C of EXP-003 (kept by the fixed decision rule); absolute crowd edges (EXP-004)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "hmm_params.json"))
    args = ap.parse_args(argv)

    dparams = calibrated_params()
    games = [("metrica_sample_game_3", load_game3()), ("skillcorner_1874553", load_match())]
    evs, labels, sizes = [], [], {}
    for name, (clean, gt, info) in games:
        tr = degrade(clean, seed=DEGRADE_SEED, params=dparams)
        evs.append(extract_evidence(tr, streams=STREAMS))
        labels.append(gt)
        sizes[name] = int(info["n_frames"])
    gt_all = np.concatenate(labels)
    params = fit(concat_evidence(evs), gt_all, gt_all != 3)
    meta = {
        "source": "pooled external games (events/phases-derived labels): " + ", ".join(sizes),
        "degrade_seed": DEGRADE_SEED,
        "degrade_params": dparams.__dict__,
        "streams": list(STREAMS),
        "n_frames": sizes,
    }
    save_params(args.out, params, meta)
    print("wrote {} (pooled {})".format(args.out, ", ".join("{}: {} frames".format(k, v) for k, v in sizes.items())))


if __name__ == "__main__":
    main()
