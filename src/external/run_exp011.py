"""EXP-011: softmax-emission temperature (0.3 reference vs 0.5 / 0.7 / 1.0), leave-one-game-out over the external games.

Protocol and rule: experiments/EXP-011.md (fixed before running). Reuses the cached features of run_exp010.

    python -m src.external.run_exp011 [--seeds 3]
"""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from ..model.frame_clf import decode_softmax, fit_softmax
from ..model.hmm import concat_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params
from .registry import available
from .run_exp003 import _score
from .run_exp010 import SWAP, log, prepare

REF = 0.3
TEMPS = {"T0.3": 0.3, "T0.5": 0.5, "T0.7": 0.7, "T1.0": 1.0}
GAIN_MIN, MIN_POS, DROP_MAX = 0.04, 5, 0.02


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    games = [prepare(n, params, args.seeds) for n in available()]
    per_game: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, held in enumerate(games):
        train = [g for j, g in enumerate(games) if j != i]
        gt_cat = np.concatenate([g["gt"] for g in train])
        nb = fit(concat_evidence([g["train_ev"] for g in train]), gt_cat, gt_cat != 3)
        model = fit_softmax(np.concatenate([g["X_sub"] for g in train] + [g["Xs_sub"] for g in train]),
                            np.concatenate([g["y_sub"] for g in train] + [SWAP[g["y_sub"]] for g in train]))
        graded = held["gt"] != 3
        acc = {v: [] for v in TEMPS}
        for X, _ in held["tests"]:
            for v, t in TEMPS.items():
                pred = decode_softmax(model, X, nb.log_trans, nb.log_prior, held["runs"], t)
                r = _score(pm, held["gt"], pred)
                r["dshare"] = float((pred[graded] == 2).mean())
                acc[v].append(r)
        per_game[held["name"]] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in TEMPS}
        log("fold {:22s} ".format(held["name"]) + " ".join("{} {:.3f}".format(v, per_game[held["name"]][v]["spell"]) for v in TEMPS))

    log("\n== spellF1 / macroF1 / D-F1 / predicted D share per held-out game ==")
    for name, d in per_game.items():
        log("{:22s} ".format(name) + "  ".join("{} {:.3f}/{:.3f}/{:.3f}/{:.2f}".format(
            v, d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["dshare"]) for v in TEMPS))
    log("mean: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(
        v, *[float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")]) for v in TEMPS))

    log("\n== decision rule vs T0.3: mean dSpell > {:.2f}, positive in >= {} of {} games, D-F1 drop <= {:.2f}, "
        "macroF1 drop <= {:.2f} ==".format(GAIN_MIN, MIN_POS, len(per_game), DROP_MAX, DROP_MAX))
    passing = []
    for v in TEMPS:
        if v == "T0.3":
            continue
        gains = np.array([d[v]["spell"] - d["T0.3"]["spell"] for d in per_game.values()])
        d_df = float(np.mean([d[v]["dF1"] - d["T0.3"]["dF1"] for d in per_game.values()]))
        d_ma = float(np.mean([d[v]["macro"] - d["T0.3"]["macro"] for d in per_game.values()]))
        ok = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_POS and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
        log("{}: {}  mean dSpell {:+.3f} (positive in {}/{}), dD-F1 {:+.3f}, dMacro {:+.3f}".format(
            v, "PASS" if ok else "fail", gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
        if ok:
            passing.append((float(gains.mean()), v))
    log("decision: {}".format("ADOPT " + max(passing)[1] if passing else "KEEP T0.3"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
