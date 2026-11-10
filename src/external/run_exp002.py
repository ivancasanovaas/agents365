"""EXP-002: HMM/Viterbi vs smooth50 on the external bench, with 2-fold cross-fitting by half.

Parameters are fitted on one half's labels and decoded on the other (and vice versa), so no frame is
ever decoded by a model that saw its labels. Aggregates only.

Usage: python -m src.external.run_exp002 --seeds 3
"""
from __future__ import annotations

import argparse

import numpy as np

from ..model.hmm import decode, extract_evidence, fit
from .baseline import nearest_player_codes, smooth_codes
from .common import load_possession_metrics
from .degrade import DegradeParams, degrade
from .metrica import load_game3


def cross_fit_predict(evidence, gt, runs, temp, use_speed, switch_scale):
    pred = np.full(len(gt), 3, dtype=np.int8)
    for k, test_run in enumerate(runs):
        train_mask = np.zeros(len(gt), dtype=bool)
        for j, (a, b) in enumerate(runs):
            if j != k:
                train_mask[a:b] = True
        params = fit(evidence, gt, train_mask & (gt != 3), switch_scale=switch_scale)
        part = decode(evidence, params, [test_run], temp=temp, use_speed=use_speed)
        a, b = test_run
        pred[a:b] = part[a:b]
    return pred


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["metrica_game3", "skillcorner"], default="metrica_game3")
    ap.add_argument("--transfer", action="store_true",
                    help="fit on metrica_game3, decode --source (true cross-game test)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.3, 0.2])
    ap.add_argument("--switch-scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    pm = load_possession_metrics()
    if args.source == "skillcorner":
        from .skillcorner import load_match
        clean, gt, info = load_match()
    else:
        clean, gt, info = load_game3()
    runs = info["halves"]  # folds = the real periods
    print("{} | frames {} | ball missing {:.3f} | label share H/A/D/X {} | runs {} | seeds {}".format(
        args.source, info["n_frames"], info["ball_missing_rate"],
        [round(float((gt == c).mean()), 3) for c in (0, 1, 2, 3)], runs, args.seeds))

    train_ev = train_gt = None
    if args.transfer:
        # TRANSFER: parameters come from Metrica Game 3 only; the target game is never used to fit.
        src_clean, train_gt, _ = load_game3()
        train_ev = extract_evidence(degrade(src_clean, seed=999, params=DegradeParams()))
        print("transfer mode: HMM fitted on metrica_game3, decoded on {}".format(args.source))

    results = {}
    for seed in range(args.seeds):
        tr = degrade(clean, seed=1000 + seed, params=DegradeParams())
        ev = extract_evidence(tr)
        variants = {"smooth50": smooth_codes(nearest_player_codes(tr), 50, "D")}
        for temp in args.temps:
            for use_speed in (False, True):
                name = "hmm temp={} {}".format(temp, "team+speed" if use_speed else "team")
                if args.transfer:
                    params = fit(train_ev, train_gt, train_gt != 3, switch_scale=args.switch_scale)
                    variants[name] = decode(ev, params, runs, temp=temp, use_speed=use_speed)
                else:
                    variants[name] = cross_fit_predict(ev, gt, runs, temp, use_speed, args.switch_scale)
        for name, pred in variants.items():
            r = pm.evaluate(gt, pred, bootstrap_n=0)
            results.setdefault(name, []).append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_segments"]))

    print("{:28s} {:>8s} {:>8s} {:>8s} {:>9s}".format("variant", "spellF1", "+-", "macroF1", "segments"))
    for name, vals in results.items():
        a = np.array(vals)
        print("{:28s} {:8.4f} {:8.4f} {:8.4f} {:9.0f}".format(
            name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean()))
    print("true segments in GT: {}".format(pm.evaluate(gt, gt.copy(), bootstrap_n=0)["n_gt_segments"]))


if __name__ == "__main__":
    main()
