"""EXP-017: train the MLP emissions also on the DFL games (native labels). Protocol and rule: experiments/EXP-017.md.

    python -m src.external.run_exp017 [--seeds 3]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import numpy as np

from ..model.frame_clf import TRAIN_STRIDE, build_features, decode_softmax, feature_names, fit_mlp
from ..model.hmm import concat_evidence, fit
from ..model.postfilter import min_duration_filter, spell_lengths
from .common import FPS, REPO_ROOT, load_possession_metrics
from .degrade import calibrated_params, degrade
from .dfl import load_match, match_ids
from .registry import available
from .run_exp003 import TEMP, _score
from .run_exp010 import SWAP, log, prepare

CACHE_DIR = REPO_ROOT / "ext_open" / "cache"
SENS_SECONDS = 3.0
DFL_TOL_DF1, DFL_TOL_TEAM, DEV_TOL_SPELL, DEV_TOL = 0.005, 0.005, 0.01, 0.02


def prepare_dfl(mid, params, n_seeds):
    key = hashlib.sha1(repr((sorted(params.__dict__.items()), n_seeds, list(feature_names(False)), TRAIN_STRIDE, "dfl1")).encode()).hexdigest()[:12]
    path = CACHE_DIR / "dfl_{}_{}.npz".format(mid, key)
    t0 = time.time()
    clean, gt, info = load_match(mid)
    runs = info["halves"]
    if path.exists():
        z = np.load(path)
        game = {"name": mid, "gt": gt, "runs": runs, "X_sub": z["X_sub"], "Xs_sub": z["Xs_sub"], "y_sub": z["y_sub"],
                "tests": [z["t{}".format(i)] for i in range(n_seeds)]}
        log("loaded DFL from cache {} [{:.0f}s]".format(mid, time.time() - t0))
        return game
    tr = degrade(clean, seed=999, params=params)
    X, Xs = build_features(tr, runs, swap_too=True)
    idx = np.flatnonzero(gt < 3)[::TRAIN_STRIDE]
    game = {"name": mid, "gt": gt, "runs": runs, "X_sub": X[idx], "Xs_sub": Xs[idx], "y_sub": gt[idx].astype(np.int64), "tests": []}
    for s in range(n_seeds):
        Xt, _ = build_features(degrade(clean, seed=3000 + s, params=params), runs)
        game["tests"].append(Xt)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_sub=game["X_sub"], Xs_sub=game["Xs_sub"], y_sub=game["y_sub"],
                        **{"t{}".format(i): x for i, x in enumerate(game["tests"])})
    log("prepared DFL {} [{:.0f}s]".format(mid, time.time() - t0))
    return game


def _metrics(pm, gt, pred, runs, gt_sens):
    r = _score(pm, gt, pred)
    graded = gt != 3
    m = (gt == 0) | (gt == 1)
    at, ap = (gt <= 1) & graded, (pred <= 1) & graded
    tp = float((at & ap).sum())
    prec, rec = tp / max(ap.sum(), 1), tp / max(at.sum(), 1)
    r["ad_f1"] = 2 * prec * rec / max(prec + rec, 1e-12)
    r["team_acc"] = float((pred[m] == gt[m]).mean())
    r["dshare_err"] = abs(float((pred[graded] == 2).mean()) - float((gt[graded] == 2).mean()))
    r["spell_sens"] = pm.evaluate(gt_sens, pred, bootstrap_n=0)["spell_f1_mean"]
    return r


def _rows(games):
    return (np.concatenate([g["X_sub"] for g in games] + [g["Xs_sub"] for g in games]),
            np.concatenate([g["y_sub"] for g in games] + [SWAP[g["y_sub"]] for g in games]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    params, pm = calibrated_params(), load_possession_metrics()
    dev = [prepare(n, params, args.seeds) for n in available()]
    dfl = [prepare_dfl(m, params, args.seeds) for m in match_ids()]
    log("{} development games, {} DFL games".format(len(dev), len(dfl)))
    for g in dfl:
        g["gt_sens"] = min_duration_filter(g["gt"], int(round(SENS_SECONDS * FPS)), g["runs"])

    def transitions(train_dev):
        gt_cat = np.concatenate([g["gt"] for g in train_dev])
        return fit(concat_evidence([g["train_ev"] for g in train_dev]), gt_cat, gt_cat != 3)

    res = {"dfl": {}, "dev": {}}
    # --- DFL folds
    nb_all = transitions(dev)
    model_dev = fit_mlp(*_rows(dev))
    log("trained dev-only model")
    for i, held in enumerate(dfl):
        t0 = time.time()
        model_b = fit_mlp(*_rows(dev + [g for j, g in enumerate(dfl) if j != i]))
        out = {}
        for tag, model in (("dev", model_dev), ("dev+DFL", model_b)):
            rs = [_metrics(pm, held["gt"], decode_softmax(model, X, nb_all.log_trans, nb_all.log_prior, held["runs"], TEMP),
                           held["runs"], held["gt_sens"]) for X in held["tests"]]
            out[tag] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        res["dfl"][held["name"]] = out
        log("DFL fold {} [{:.0f}s] spellF1 dev {:.3f} dev+DFL {:.3f} | D-F1 {:.3f} {:.3f} | team acc {:.3f} {:.3f}".format(
            held["name"], time.time() - t0, out["dev"]["spell"], out["dev+DFL"]["spell"], out["dev"]["dF1"],
            out["dev+DFL"]["dF1"], out["dev"]["team_acc"], out["dev+DFL"]["team_acc"]))
    # --- development folds
    for i, held in enumerate(dev):
        t0 = time.time()
        train_dev = [g for j, g in enumerate(dev) if j != i]
        nb = transitions(train_dev)
        out = {}
        for tag, train in (("dev", train_dev), ("dev+DFL", train_dev + dfl)):
            model = fit_mlp(*_rows(train))
            rs = [_metrics(pm, held["gt"], decode_softmax(model, X, nb.log_trans, nb.log_prior, held["runs"], TEMP),
                           held["runs"], held["gt"]) for X, _ in held["tests"]]
            out[tag] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        res["dev"][held["name"]] = out
        log("dev fold {} [{:.0f}s] spellF1 dev {:.3f} dev+DFL {:.3f}".format(
            held["name"], time.time() - t0, out["dev"]["spell"], out["dev+DFL"]["spell"]))

    def mean(group, tag, key):
        return float(np.mean([d[tag][key] for d in res[group].values()]))

    log("\n== DFL folds (mean): D-F1 / alive-dead F1 / team acc / |D share error| / spellF1 / spellF1 vs 3 s-merged ==")
    for tag in ("dev", "dev+DFL"):
        log("{:8s} {:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.3f}".format(
            tag, *[mean("dfl", tag, k) for k in ("dF1", "ad_f1", "team_acc", "dshare_err", "spell", "spell_sens")]))
    log("== development folds (mean): spellF1 / macroF1 / D-F1 ==")
    for tag in ("dev", "dev+DFL"):
        log("{:8s} {:.3f} / {:.3f} / {:.3f}".format(tag, *[mean("dev", tag, k) for k in ("spell", "macro", "dF1")]))

    c1 = (mean("dfl", "dev+DFL", "dF1") >= mean("dfl", "dev", "dF1") - DFL_TOL_DF1
          and mean("dfl", "dev+DFL", "dshare_err") <= mean("dfl", "dev", "dshare_err")
          and mean("dfl", "dev+DFL", "team_acc") >= mean("dfl", "dev", "team_acc") - DFL_TOL_TEAM)
    c2 = (mean("dev", "dev+DFL", "spell") >= mean("dev", "dev", "spell") - DEV_TOL_SPELL
          and mean("dev", "dev+DFL", "macro") >= mean("dev", "dev", "macro") - DEV_TOL
          and mean("dev", "dev+DFL", "dF1") >= mean("dev", "dev", "dF1") - DEV_TOL)
    log("\n== rule: (1) DFL folds {} | (2) development folds no-harm {} ==".format("PASS" if c1 else "fail", "PASS" if c2 else "fail"))
    log("decision: {}".format("ADOPT dev+DFL" if (c1 and c2) else "KEEP dev"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
