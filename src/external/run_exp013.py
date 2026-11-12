"""EXP-013: one-hidden-layer MLP emissions vs the linear softmax, leave-one-game-out over the external games.

Protocol and rule: experiments/EXP-013.md (fixed before running). Reuses the cached base features of run_exp010.

    python -m src.external.run_exp013 [--seeds 3]
"""
from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from ..model.frame_clf import decode_softmax, fit_mlp, fit_softmax
from ..model.hmm import concat_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params
from .registry import available
from .run_exp003 import TEMP, _score
from .run_exp010 import SWAP, log, prepare

GAIN_MIN, MIN_POS, DROP_MAX = 0.04, 5, 0.02
VARIANTS = ("softmax", "mlp")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    games = [prepare(n, params, args.seeds) for n in available()]
    per_game: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, held in enumerate(games):
        t0 = time.time()
        train = [g for j, g in enumerate(games) if j != i]
        gt_cat = np.concatenate([g["gt"] for g in train])
        nb = fit(concat_evidence([g["train_ev"] for g in train]), gt_cat, gt_cat != 3)
        Xtr = np.concatenate([g["X_sub"] for g in train] + [g["Xs_sub"] for g in train])
        ytr = np.concatenate([g["y_sub"] for g in train] + [SWAP[g["y_sub"]] for g in train])
        models = {"softmax": fit_softmax(Xtr, ytr), "mlp": fit_mlp(Xtr, ytr)}
        graded = held["gt"] != 3
        acc = {v: [] for v in VARIANTS}
        for X, _ in held["tests"]:
            for v in VARIANTS:
                pred = decode_softmax(models[v], X, nb.log_trans, nb.log_prior, held["runs"], TEMP)
                r = _score(pm, held["gt"], pred)
                r["dshare"] = float((pred[graded] == 2).mean())
                acc[v].append(r)
        per_game[held["name"]] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in VARIANTS}
        log("fold {:22s} [{:.0f}s] softmax {:.3f} mlp {:.3f}".format(
            held["name"], time.time() - t0, per_game[held["name"]]["softmax"]["spell"], per_game[held["name"]]["mlp"]["spell"]))

    log("\n== spellF1 / macroF1 / D-F1 / predicted D share per held-out game ==")
    for name, d in per_game.items():
        log("{:22s} ".format(name) + "  ".join("{} {:.3f}/{:.3f}/{:.3f}/{:.2f}".format(
            v, d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["dshare"]) for v in VARIANTS))
    log("mean: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(
        v, *[float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")]) for v in VARIANTS))

    gains = np.array([d["mlp"]["spell"] - d["softmax"]["spell"] for d in per_game.values()])
    d_df = float(np.mean([d["mlp"]["dF1"] - d["softmax"]["dF1"] for d in per_game.values()]))
    d_ma = float(np.mean([d["mlp"]["macro"] - d["softmax"]["macro"] for d in per_game.values()]))
    ok = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_POS and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
    log("\n== decision rule vs softmax: mean dSpell > {:.2f}, positive in >= {} of {}, D-F1 drop <= {:.2f}, macro drop <= {:.2f} ==".format(
        GAIN_MIN, MIN_POS, len(gains), DROP_MAX, DROP_MAX))
    log("mlp: {}  mean dSpell {:+.3f} (positive in {}/{}), dD-F1 {:+.3f}, dMacro {:+.3f}".format(
        "PASS" if ok else "fail", gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
    log("decision: {}".format("ADOPT mlp" if ok else "KEEP softmax"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
