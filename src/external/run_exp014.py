"""EXP-014: untouched hold-out games (unbiased estimate) and confirmation of the MLP.

Models are trained as in src/fit_params.py on the DEVELOPMENT games (registry.available()), then scored once on the hold-out
games (registry.holdout_available()). Protocol and rule: experiments/EXP-014.md (fixed before the hold-out was loaded).

    python -m src.external.run_exp014 [--seeds 3]
"""
from __future__ import annotations

import argparse

import numpy as np

from ..model.frame_clf import build_features, decode_softmax, fit_mlp, fit_softmax
from ..model.hmm import concat_evidence, decode, extract_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params, degrade
from .registry import available, holdout_available, load_holdout
from .run_exp003 import TEMP, _score
from .run_exp010 import NB_STREAMS, SWAP, log, prepare

VARIANTS = ("A", "C", "softmax", "mlp")
MIN_POS, GAIN_MIN, DROP_MAX = 3, 0.02, 0.02


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    dev = [prepare(n, params, args.seeds) for n in available()]
    gt_cat = np.concatenate([g["gt"] for g in dev])
    nb = fit(concat_evidence([g["train_ev"] for g in dev]), gt_cat, gt_cat != 3)
    Xtr = np.concatenate([g["X_sub"] for g in dev] + [g["Xs_sub"] for g in dev])
    ytr = np.concatenate([g["y_sub"] for g in dev] + [SWAP[g["y_sub"]] for g in dev])
    models = {"softmax": fit_softmax(Xtr, ytr), "mlp": fit_mlp(Xtr, ytr)}
    log("trained on {} development games, {} rows".format(len(dev), len(ytr)))

    names = holdout_available()
    if not names:
        raise SystemExit("hold-out games not found under ext_open/ (python scripts/fetch_external.py)")
    per_game = {}
    for name in names:
        clean, gt, info = load_holdout(name)
        runs, graded = info["halves"], gt != 3
        acc = {v: [] for v in VARIANTS}
        for s in range(args.seeds):
            tr = degrade(clean, seed=3000 + s, params=params)
            X, _ = build_features(tr, runs)
            ev = extract_evidence(tr, streams=NB_STREAMS)
            preds = {"A": decode(ev, nb, runs, temp=TEMP, use_speed=False, streams=()),
                     "C": decode(ev, nb, runs, temp=TEMP, use_speed=False, streams=NB_STREAMS)}
            for v in ("softmax", "mlp"):
                preds[v] = decode_softmax(models[v], X, nb.log_trans, nb.log_prior, runs, TEMP)
            for v in VARIANTS:
                r = _score(pm, gt, preds[v])
                r["dshare"] = float((preds[v][graded] == 2).mean())
                acc[v].append(r)
        per_game[name] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in VARIANTS}
        per_game[name]["true_d"] = float((gt[graded] == 2).mean())
        log("scored {:22s} spellF1 A {:.3f} C {:.3f} softmax {:.3f} mlp {:.3f}".format(
            name, *[per_game[name][v]["spell"] for v in VARIANTS]))

    log("\n== hold-out: spellF1 / macroF1 / D-F1 / predicted D share (true D share) ==")
    for name, d in per_game.items():
        log("{:22s} ".format(name) + "  ".join("{} {:.3f}/{:.3f}/{:.3f}/{:.2f}".format(
            v, d[v]["spell"], d[v]["macro"], d[v]["dF1"], d[v]["dshare"]) for v in VARIANTS) + "  (true D {:.2f})".format(d["true_d"]))
    mean = {v: {k: float(np.mean([d[v][k] for d in per_game.values()])) for k in ("spell", "macro", "dF1")} for v in VARIANTS}
    log("mean over hold-out: " + " | ".join("{} {:.3f}/{:.3f}/{:.3f}".format(
        v, mean[v]["spell"], mean[v]["macro"], mean[v]["dF1"]) for v in VARIANTS))
    log("softmax vs C: mean dSpell {:+.3f}, vs A: {:+.3f}".format(
        mean["softmax"]["spell"] - mean["C"]["spell"], mean["softmax"]["spell"] - mean["A"]["spell"]))

    gains = np.array([d["mlp"]["spell"] - d["softmax"]["spell"] for d in per_game.values()])
    d_df = mean["mlp"]["dF1"] - mean["softmax"]["dF1"]
    d_ma = mean["mlp"]["macro"] - mean["softmax"]["macro"]
    ok = int((gains > 0).sum()) >= MIN_POS and gains.mean() >= GAIN_MIN and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
    log("\n== MLP rule: higher spellF1 in >= {} of {} games, mean gain >= {:.2f}, D-F1 and macroF1 drops <= {:.2f} ==".format(
        MIN_POS, len(gains), GAIN_MIN, DROP_MAX))
    log("mlp vs softmax: {}  mean dSpell {:+.3f} (higher in {}/{}), dD-F1 {:+.3f}, dMacro {:+.3f}".format(
        "PASS" if ok else "fail", gains.mean(), int((gains > 0).sum()), len(gains), d_df, d_ma))
    log("decision: {}".format("ADOPT mlp" if ok else "KEEP softmax"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
