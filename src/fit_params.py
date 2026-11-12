"""Fit the possession model on EXTERNAL games only and save its parameters (no competition data involved).

    python -m src.fit_params            # needs the games under ext_open/ (sources in PROVENANCE.md)

Estimated on ALL available external games (Metrica Sample Games 1-3, SkillCorner open matches), each degraded with the
profile-calibrated bench (seed 999):
  - models/hmm_params.json      transition matrix and prior by counting (+ the naive-Bayes streams kept for reference);
  - models/softmax_params.json  linear softmax emission model on frame features (EXP-010), the reference;
  - models/mlp_params.json      one-hidden-layer MLP emission model (EXP-013/014), the PRODUCTION model. Both are trained on
                                every 5th graded frame and its team-swapped copy.
Leave-one-game-out validation of this exact recipe: python -m src.external.run_exp010.
"""
from __future__ import annotations

import argparse

import numpy as np

from .external.common import REPO_ROOT
from .external.degrade import calibrated_params, degrade
from .external.registry import available, load
from .model.frame_clf import TRAIN_STRIDE, build_features, fit_mlp, fit_softmax
from .model.hmm import concat_evidence, extract_evidence, fit
from .model.io import save_mlp, save_params, save_softmax

DEGRADE_SEED = 999
STREAMS = ("out", "crowd")      # naive-Bayes streams (variant C); the production emissions are the softmax model
SWAP = np.array([1, 0, 2])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmm-out", default=str(REPO_ROOT / "models" / "hmm_params.json"))
    ap.add_argument("--softmax-out", default=str(REPO_ROOT / "models" / "softmax_params.json"))
    ap.add_argument("--mlp-out", default=str(REPO_ROOT / "models" / "mlp_params.json"))
    args = ap.parse_args(argv)

    dparams = calibrated_params()
    names = available()
    if len(names) < 2:
        raise SystemExit("need the external games under ext_open/ (see PROVENANCE.md); found: {}".format(names))
    evs, gts, xs, xss, ys, sizes = [], [], [], [], [], {}
    for name in names:
        clean, gt, info = load(name)
        tr = degrade(clean, seed=DEGRADE_SEED, params=dparams)
        X, Xs = build_features(tr, info["halves"], swap_too=True)
        idx = np.flatnonzero(gt < 3)[::TRAIN_STRIDE]
        evs.append(extract_evidence(tr, streams=STREAMS))
        gts.append(gt)
        xs.append(X[idx]); xss.append(Xs[idx]); ys.append(gt[idx].astype(np.int64))
        sizes[name] = int(info["n_frames"])
        print("prepared {} ({} frames)".format(name, info["n_frames"]))

    gt_all = np.concatenate(gts)
    hmm = fit(concat_evidence(evs), gt_all, gt_all != 3)
    meta = {"source": "pooled external games (events/phases-derived labels): " + ", ".join(sizes),
            "degrade_seed": DEGRADE_SEED, "degrade_params": dparams.__dict__, "streams": list(STREAMS),
            "n_frames": sizes}
    save_params(args.hmm_out, hmm, meta)

    Xt = np.concatenate(xs + xss)
    yt = np.concatenate(ys + [SWAP[y] for y in ys])
    train_meta = dict(meta, train_rows=int(len(yt)), stride=TRAIN_STRIDE)
    save_softmax(args.softmax_out, fit_softmax(Xt, yt), train_meta)              # kept as the linear reference
    save_mlp(args.mlp_out, fit_mlp(Xt, yt), train_meta)                          # production model (EXP-014)
    print("wrote {}, {} and {} ({} games, {} training rows)".format(
        args.hmm_out, args.softmax_out, args.mlp_out, len(names), len(yt)))


if __name__ == "__main__":
    main()
