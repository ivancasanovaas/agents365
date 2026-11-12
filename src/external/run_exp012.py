"""EXP-012: temporal-context features for the softmax emissions, leave-one-game-out over the external games.

Protocol and rule: experiments/EXP-012.md (fixed before running). Features are cached under ext_open/cache.

    python -m src.external.run_exp012 [--seeds 3]
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from ..model.frame_clf import CTX_GAP_NAMES, CTX_WIN_NAMES, FEATURE_NAMES, decode_softmax, feature_names, fit_softmax
from ..model.hmm import concat_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params
from .registry import available
from .run_exp003 import TEMP, _score
from .run_exp010 import SWAP, log, prepare

NAMES = feature_names(True)
COLS = {
    "base": [NAMES.index(n) for n in FEATURE_NAMES],
    "+win": [NAMES.index(n) for n in FEATURE_NAMES + CTX_WIN_NAMES],
    "+gap": [NAMES.index(n) for n in FEATURE_NAMES + CTX_GAP_NAMES],
    "ctx": list(range(len(NAMES))),
}
GAIN_MIN, MIN_POS, DROP_MAX = 0.04, 5, 0.02


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    games = [prepare(n, params, args.seeds, context=True) for n in available()]
    per_game: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, held in enumerate(games):
        train = [g for j, g in enumerate(games) if j != i]
        gt_cat = np.concatenate([g["gt"] for g in train])
        nb = fit(concat_evidence([g["train_ev"] for g in train]), gt_cat, gt_cat != 3)
        Xtr = np.concatenate([g["X_sub"] for g in train] + [g["Xs_sub"] for g in train])
        ytr = np.concatenate([g["y_sub"] for g in train] + [SWAP[g["y_sub"]] for g in train])
        models = {v: fit_softmax(Xtr[:, cols], ytr) for v, cols in COLS.items()}
        graded = held["gt"] != 3
        acc: Dict[str, List[Dict[str, float]]] = {v: [] for v in COLS}
        for X, _ in held["tests"]:
            for v, cols in COLS.items():
                pred = decode_softmax(models[v], X[:, cols], nb.log_trans, nb.log_prior, held["runs"], TEMP)
                r = _score(pm, held["gt"], pred)
                r["dshare"] = float((pred[graded] == 2).mean())
                acc[v].append(r)
        per_game[held["name"]] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in COLS}
        log("fold {:22s} ".format(held["name"]) + " ".join("{} {:.3f}".format(v, per_game[held["name"]][v]["spell"]) for v in COLS))

    log("\n== spellF1 / macroF1 / D-F1 / predicted D share per held-out game ==")
    for name, d in per_game.items():
        log("{:22s} ".format(name) + "  ".join("{} {:.3f}/{:.3f}/{:.3f}/{:.2f}".format(
            v, d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["dshare"]) for v in COLS))
    log("mean: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(
        v, *[float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")]) for v in COLS))

    log("\n== vs base: mean dSpell, games positive, dD-F1, dMacro (rule applies to ctx only) ==")
    for v in COLS:
        if v == "base":
            continue
        gains = np.array([d[v]["spell"] - d["base"]["spell"] for d in per_game.values()])
        d_df = float(np.mean([d[v]["dF1"] - d["base"]["dF1"] for d in per_game.values()]))
        d_ma = float(np.mean([d[v]["macro"] - d["base"]["macro"] for d in per_game.values()]))
        ok = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_POS and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
        log("{}: {}{}  mean dSpell {:+.3f} (positive in {}/{}), dD-F1 {:+.3f}, dMacro {:+.3f}".format(
            v, "PASS" if ok else "fail", "" if v == "ctx" else " (ablation, not decisive)",
            gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
    gains = np.array([d["ctx"]["spell"] - d["base"]["spell"] for d in per_game.values()])
    d_df = float(np.mean([d["ctx"]["dF1"] - d["base"]["dF1"] for d in per_game.values()]))
    d_ma = float(np.mean([d["ctx"]["macro"] - d["base"]["macro"] for d in per_game.values()]))
    adopt = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_POS and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
    log("decision: {}".format("ADOPT ctx" if adopt else "KEEP base"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
