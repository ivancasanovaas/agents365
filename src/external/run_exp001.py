"""EXP-001 on the external bench: smooth50 with no-ball -> D (A) vs -> X (B).

Usage:  python -m src.external.run_exp001 --source synthetic --seeds 5
Prints aggregates only. Never point this at competition data.
"""
from __future__ import annotations

import argparse

import numpy as np

from .baseline import nearest_player_codes, smooth_codes
from .common import load_possession_metrics
from .degrade import DegradeParams, degrade
from .synthetic import make_match

NOISE_BAND = 0.04  # possession_metrics.SPELL_NOISE_BAND


def evaluate_variant(pm, gt, pred, bootstrap_n=0):
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "metrica_game3"], default="synthetic")
    ap.add_argument("--no-degrade", action="store_true", help="evaluate on the clean external tracking")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--p-ball-missing", type=float, default=DegradeParams.p_ball_missing)
    args = ap.parse_args(argv)

    pm = load_possession_metrics()
    params = DegradeParams(p_ball_missing=args.p_ball_missing)
    rows = {"A_no_ball_D": [], "B_no_ball_X": []}
    fixed = None
    if args.source == "metrica_game3":
        from .metrica import load_game3
        clean, gt, info = load_game3()
        fixed = (clean, gt)
        print("loaded metrica_game3: {}".format(info))
        print("label share H/A/D: {}".format(
            [round(float((gt == c).mean()), 3) for c in (0, 1, 2)]))
    for seed in range(args.seeds):
        clean, gt = fixed if fixed else make_match(seed)
        tr = clean if args.no_degrade else degrade(clean, seed=1000 + seed, params=params)
        codes = nearest_player_codes(tr)
        for name, mode in (("A_no_ball_D", "D"), ("B_no_ball_X", "X")):
            res = evaluate_variant(pm, gt, smooth_codes(codes, args.window, mode))
            rows[name].append((res["spell_f1_mean"], res["macro_f1"]))

    print("source={} seeds={} window={} p_ball_missing={}".format(
        args.source, args.seeds, args.window, params.p_ball_missing))
    means = {}
    for name, vals in rows.items():
        arr = np.array(vals)
        means[name] = arr.mean(0)
        print("{:12s} spellF1 {:.4f} (+-{:.4f})  macroF1 {:.4f}".format(
            name, arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean()))
    gain = means["B_no_ball_X"][0] - means["A_no_ball_D"][0]
    verdict = "B better" if gain > NOISE_BAND else ("A better" if gain < -NOISE_BAND else "within noise")
    print("spellF1 gain B-A = {:+.4f} -> {} (noise band {})".format(gain, verdict, NOISE_BAND))


if __name__ == "__main__":
    main()
