"""EXP-015: 4-state HMM (dead ball remembers the last possessor) vs the 3-state HMM, with the production MLP emissions,
leave-one-game-out over the development games. Protocol and rule: experiments/EXP-015.md (fixed before running).

    python -m src.external.run_exp015 [--seeds 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..model.frame_clf import decode_softmax, fit_mlp, predict_log_proba
from ..model.hmm import concat_evidence, fit
from ..model.hmm4 import decode4, fit_transitions4
from .common import load_possession_metrics
from .degrade import calibrated_params
from .registry import available
from .run_exp003 import TEMP, _score
from .run_exp010 import SWAP, log, prepare

GAIN_MIN, MIN_POS, DROP_MAX = 0.04, 5, 0.02
VARIANTS = ("3-state", "4-state")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    games = [prepare(n, params, args.seeds) for n in available()]
    per_game = {}
    for i, held in enumerate(games):
        t0 = time.time()
        train = [g for j, g in enumerate(games) if j != i]
        gt_cat = np.concatenate([g["gt"] for g in train])
        nb = fit(concat_evidence([g["train_ev"] for g in train]), gt_cat, gt_cat != 3)
        lt4, lp4 = fit_transitions4([g["gt"] for g in train], [g["runs"] for g in train])
        model = fit_mlp(np.concatenate([g["X_sub"] for g in train] + [g["Xs_sub"] for g in train]),
                        np.concatenate([g["y_sub"] for g in train] + [SWAP[g["y_sub"]] for g in train]))
        graded = held["gt"] != 3
        acc = {v: [] for v in VARIANTS}
        for X, _ in held["tests"]:
            preds = {"3-state": decode_softmax(model, X, nb.log_trans, nb.log_prior, held["runs"], TEMP)}
            ll3 = TEMP * (predict_log_proba(model, X) - model.log_prior[None, :])
            preds["4-state"] = decode4(ll3, lt4, lp4, held["runs"])
            for v in VARIANTS:
                r = _score(pm, held["gt"], preds[v])
                r["dshare"] = float((preds[v][graded] == 2).mean())
                acc[v].append(r)
        per_game[held["name"]] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in VARIANTS}
        log("fold {:22s} [{:.0f}s] 3-state {:.3f} 4-state {:.3f}".format(
            held["name"], time.time() - t0, per_game[held["name"]]["3-state"]["spell"], per_game[held["name"]]["4-state"]["spell"]))

    log("\n== spellF1 / macroF1 / D-F1 / segments per held-out game ==")
    for name, d in per_game.items():
        log("{:22s} ".format(name) + "  ".join("{} {:.3f}/{:.3f}/{:.3f}/{:.0f}".format(
            v, d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["segs"]) for v in VARIANTS))
    mean = {v: {k: float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")} for v in VARIANTS}
    log("mean: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(v, mean[v]["spell"], mean[v]["macro"], mean[v]["dF1"]) for v in VARIANTS))

    gains = np.array([d["4-state"]["spell"] - d["3-state"]["spell"] for d in per_game.values()])
    d_df, d_ma = mean["4-state"]["dF1"] - mean["3-state"]["dF1"], mean["4-state"]["macro"] - mean["3-state"]["macro"]
    ok = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_POS and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
    log("\n== rule: mean dSpell > {:.2f}, positive in >= {} of {}, D-F1 and macroF1 drops <= {:.2f} ==".format(
        GAIN_MIN, MIN_POS, len(gains), DROP_MAX))
    log("4-state vs 3-state: {}  mean dSpell {:+.3f} (positive in {}/{}), dD-F1 {:+.3f}, dMacro {:+.3f}".format(
        "PASS" if ok else "fail", gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
    log("decision: {}".format("ADOPT 4-state" if ok else "KEEP 3-state"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
