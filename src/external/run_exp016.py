"""EXP-016: fresh cross-provider evaluation of the production artefacts on the Sportec Open DFL games.

Protocol: experiments/EXP-016.md (fixed before any DFL game was scored). Nothing is selected here.

    python -m src.external.run_exp016 [--seeds 3] [--matches DFL-MAT-J03WMX ...]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ..model.frame_clf import build_features, decode_softmax
from ..model.hmm import decode, extract_evidence
from ..model.io import load_mlp, load_params, load_softmax
from ..model.postfilter import min_duration_filter, spell_lengths
from .common import FPS, REPO_ROOT, load_possession_metrics
from .degrade import calibrated_params, degrade
from .dfl import load_match, match_ids
from .run_exp003 import TEMP, _score

STREAMS = ("out", "crowd")
VARIANTS = ("A", "C", "softmax", "mlp")
SENS_SECONDS = 3.0           # the DFL median spell: sensitivity only, not tuned


def _team_acc(gt, pred):
    m = (gt == 0) | (gt == 1)
    return float((pred[m] == gt[m]).mean())


def _ad_f1(gt, pred):
    g, p = gt != 3, None
    alive_true, alive_pred = (gt <= 1) & g, (pred <= 1) & g
    tp = float((alive_true & alive_pred).sum())
    prec, rec = tp / max(alive_pred.sum(), 1), tp / max(alive_true.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def log(msg):
    print(msg)
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    models = REPO_ROOT / "models"
    nb, _ = load_params(models / "hmm_params.json")
    soft, _ = load_softmax(models / "softmax_params.json")
    mlp, _ = load_mlp(models / "mlp_params.json")

    ids = args.matches or match_ids()
    per = {}
    for mid in ids:
        clean, gt, info = load_match(mid)
        runs, graded = info["halves"], gt != 3
        true_spells = len(spell_lengths(gt, runs))
        gt_sens = min_duration_filter(gt, int(round(SENS_SECONDS * FPS)), runs)
        acc = {v: [] for v in VARIANTS}
        for s in range(args.seeds):
            tr = degrade(clean, seed=3000 + s, params=params)
            X, _ = build_features(tr, runs)
            ev = extract_evidence(tr, streams=STREAMS)
            preds = {"A": decode(ev, nb, runs, temp=TEMP, use_speed=False, streams=()),
                     "C": decode(ev, nb, runs, temp=TEMP, use_speed=False, streams=STREAMS),
                     "softmax": decode_softmax(soft, X, nb.log_trans, nb.log_prior, runs, TEMP),
                     "mlp": decode_softmax(mlp, X, nb.log_trans, nb.log_prior, runs, TEMP)}
            for v in VARIANTS:
                r = _score(pm, gt, preds[v])
                r["spell_sens"] = pm.evaluate(gt_sens, preds[v], bootstrap_n=0)["spell_f1_mean"]
                r["team_acc"] = _team_acc(gt, preds[v])
                r["ad_f1"] = _ad_f1(gt, preds[v])
                r["dshare"] = float((preds[v][graded] == 2).mean())
                r["pred_spells"] = float(len(spell_lengths(preds[v], runs)))
                acc[v].append(r)
        per[mid] = {v: {k: float(np.mean([r[k] for r in acc[v]])) for k in acc[v][0]} for v in VARIANTS}
        per[mid]["true_d"] = float((gt[graded] == 2).mean())
        per[mid]["true_spells"] = true_spells
        log("scored {} spellF1 A {:.3f} C {:.3f} softmax {:.3f} mlp {:.3f} | D-F1 mlp {:.3f} | team acc mlp {:.3f}".format(
            mid, *[per[mid][v]["spell"] for v in VARIANTS], per[mid]["mlp"]["dF1"], per[mid]["mlp"]["team_acc"]))

    log("\n== per match (mlp): spellF1 / spellF1 vs 3 s-merged labels / D-F1 / alive-dead F1 / team acc / pred D (true D) / spells pred (true) ==")
    for mid, d in per.items():
        m = d["mlp"]
        log("{} {:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.2f} ({:.2f}) / {:.0f} ({})".format(
            mid, m["spell"], m["spell_sens"], m["dF1"], m["ad_f1"], m["team_acc"], m["dshare"], d["true_d"],
            m["pred_spells"], d["true_spells"]))
    log("\n== mean over matches ==")
    for v in VARIANTS:
        log("{:8s} spellF1 {:.3f} | sens(3s) {:.3f} | macroF1 {:.3f} | D-F1 {:.3f} | alive/dead F1 {:.3f} | team acc {:.3f}".format(
            v, *[float(np.mean([d[v][k] for d in per.values()])) for k in
                 ("spell", "spell_sens", "macro", "dF1", "ad_f1", "team_acc")]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
