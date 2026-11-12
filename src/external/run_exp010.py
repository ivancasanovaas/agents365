"""EXP-010: discriminative (softmax) emissions vs naive-Bayes streams, leave-one-game-out over the external games.

Protocol and decision rule: experiments/EXP-010.md (fixed before running). For each held-out game the other games are
degraded (bench v2, seed 999) and used for training; the held-out game is degraded with seeds 3000..3002 and the per-game
score is the mean over those seeds. Aggregates only; no competition data.

    python -m src.external.run_exp010 [--seeds 3] [--games name1 name2 ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

import numpy as np

from ..model.frame_clf import TRAIN_STRIDE, build_features, decode_softmax, fit_softmax
from ..model.hmm import concat_evidence, decode, extract_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params, degrade
from .registry import available, load
from .run_exp003 import TEMP, _score

NB_STREAMS = ("out", "crowd")
TEMPS = {"S0.3": 0.3, "S0.1": 0.1}
VARIANTS = ("A", "C") + tuple(TEMPS)
GAIN_MIN, MIN_GAMES_POSITIVE, DROP_MAX = 0.04, 5, 0.02
SWAP = np.array([1, 0, 2])


def log(msg):
    print(msg)
    sys.stdout.flush()


def prepare(name, params, n_seeds):
    t0 = time.time()
    clean, gt, info = load(name)
    runs = info["halves"]
    tr = degrade(clean, seed=999, params=params)
    X, Xs = build_features(tr, runs, swap_too=True)
    idx = np.flatnonzero(gt < 3)[::TRAIN_STRIDE]
    game = {"name": name, "gt": gt, "runs": runs,
            "train_ev": extract_evidence(tr, streams=NB_STREAMS),
            "X_sub": X[idx], "Xs_sub": Xs[idx], "y_sub": gt[idx].astype(np.int64), "tests": []}
    del X, Xs
    for s in range(n_seeds):
        trs = degrade(clean, seed=3000 + s, params=params)
        Xt, _ = build_features(trs, runs, swap_too=False)
        game["tests"].append((Xt, extract_evidence(trs, streams=NB_STREAMS)))
    log("prepared {:22s} frames {:>7d} [{:.0f}s]".format(name, len(gt), time.time() - t0))
    return game


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--games", nargs="*", default=None)
    args = ap.parse_args(argv)

    names = args.games or available()
    log("external games: {}".format(names))
    params = calibrated_params()
    pm = load_possession_metrics()
    games = [prepare(n, params, args.seeds) for n in names]

    per_game: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, held in enumerate(games):
        t0 = time.time()
        train = [g for j, g in enumerate(games) if j != i]
        gt_cat = np.concatenate([g["gt"] for g in train])
        nb = fit(concat_evidence([g["train_ev"] for g in train]), gt_cat, gt_cat != 3)
        Xt = np.concatenate([g["X_sub"] for g in train] + [g["Xs_sub"] for g in train])
        yt = np.concatenate([g["y_sub"] for g in train] + [SWAP[g["y_sub"]] for g in train])
        model = fit_softmax(Xt, yt)
        graded = held["gt"] != 3
        acc: Dict[str, List[Dict[str, float]]] = {v: [] for v in VARIANTS}
        dshare: Dict[str, List[float]] = {v: [] for v in VARIANTS}
        for X, ev in held["tests"]:
            preds = {
                "A": decode(ev, nb, held["runs"], temp=TEMP, use_speed=False, streams=()),
                "C": decode(ev, nb, held["runs"], temp=TEMP, use_speed=False, streams=NB_STREAMS),
            }
            for v, temp in TEMPS.items():
                preds[v] = decode_softmax(model, X, nb.log_trans, nb.log_prior, held["runs"], temp)
            for v in VARIANTS:
                acc[v].append(_score(pm, held["gt"], preds[v]))
                dshare[v].append(float((preds[v][graded] == 2).mean()))
        per_game[held["name"]] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in VARIANTS}
        for v in VARIANTS:
            per_game[held["name"]][v]["dshare"] = float(np.mean(dshare[v]))
        per_game[held["name"]]["true_d"] = float((held["gt"][graded] == 2).mean())
        log("fold {:22s} done [{:.0f}s]  spellF1 A {:.3f} C {:.3f} S0.3 {:.3f} S0.1 {:.3f}".format(
            held["name"], time.time() - t0, *[per_game[held["name"]][v]["spell"] for v in VARIANTS]))

    log("\n== per held-out game: spellF1 / macroF1 / D-F1 / predicted D share (true D share) ==")
    log("{:22s} {:^22s} {:^22s} {:^22s} {:^22s}".format("game", "A", "C", "S0.3", "S0.1"))
    for name, d in per_game.items():
        cells = ["{:.3f}/{:.3f}/{:.3f}/{:.2f}".format(d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["dshare"])
                 for v in VARIANTS]
        log("{:22s} {}  (true D {:.2f})".format(name, "  ".join("{:^22s}".format(c) for c in cells), d["true_d"]))
    mean = {v: {k: float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")}
            for v in VARIANTS}
    log("\nmean over games: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(v, mean[v]["spell"], mean[v]["macro"], mean[v]["dF1"])
                                          for v in VARIANTS))

    gain_ca = [d["C"]["spell"] - d["A"]["spell"] for d in per_game.values()]
    log("reference C vs A: mean dSpell {:+.3f}, positive in {} of {} games".format(
        float(np.mean(gain_ca)), int(sum(g > 0 for g in gain_ca)), len(gain_ca)))

    log("\n== decision rule vs C: mean dSpell > {:.2f}, positive in >= {} of {} games, mean D-F1 drop <= {:.2f}, "
        "mean macroF1 drop <= {:.2f} ==".format(GAIN_MIN, MIN_GAMES_POSITIVE, len(per_game), DROP_MAX, DROP_MAX))
    passing = []
    for v in TEMPS:
        gains = np.array([d[v]["spell"] - d["C"]["spell"] for d in per_game.values()])
        d_df = float(np.mean([d[v]["dF1"] - d["C"]["dF1"] for d in per_game.values()]))
        d_ma = float(np.mean([d[v]["macro"] - d["C"]["macro"] for d in per_game.values()]))
        ok = gains.mean() > GAIN_MIN and int((gains > 0).sum()) >= MIN_GAMES_POSITIVE and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
        log("{}: {}  mean dSpell {:+.3f} (positive in {}/{}), mean dD-F1 {:+.3f}, mean dMacro {:+.3f}".format(
            v, "PASS" if ok else "fail", gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
        if ok:
            passing.append((float(gains.mean()), v))
    log("decision: {}".format("ADOPT " + max(passing)[1] if passing else "KEEP C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
