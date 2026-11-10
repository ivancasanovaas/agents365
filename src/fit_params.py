"""Fit the HMM on an EXTERNAL game and save the parameters (no competition data involved).

    python -m src.fit_params            # needs ext_open/metrica_game3 (see PROVENANCE.md for the source)

Default: Metrica Sample Game 3, the same EPTS family as the competition ATD, validated by transfer
to SkillCorner (EXP-002). Writes models/hmm_params.json.
"""
from __future__ import annotations

import argparse

from .external.common import REPO_ROOT
from .external.degrade import DegradeParams, degrade
from .external.metrica import load_game3
from .model.hmm import extract_evidence, fit
from .model.io import save_params

DEGRADE_SEED = 999


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "hmm_params.json"))
    args = ap.parse_args(argv)

    clean, gt, info = load_game3()
    tr = degrade(clean, seed=DEGRADE_SEED, params=DegradeParams())
    params = fit(extract_evidence(tr), gt, gt != 3)
    meta = {
        "source": "metrica_sample_game_3 (events-derived labels)",
        "degrade_seed": DEGRADE_SEED,
        "degrade_params": DegradeParams().__dict__,
        "n_frames": info["n_frames"],
    }
    save_params(args.out, params, meta)
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
