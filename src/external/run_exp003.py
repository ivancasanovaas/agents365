"""EXP-003: dead-ball evidence as extra HMM streams, evaluated on the external bench only.

Variants are cumulative (see experiments/EXP-003.md):
  A team only | B + ball outside pitch | C + crowd stillness | D + gap-start near a line
Settings: (1) Metrica Game 3 with 2-fold cross-fitting by half; (2) transfer: fit on Metrica, decode SkillCorner.
The decision rule is applied mechanically at the end. Aggregates only; no competition data.

    python -m src.external.run_exp003 --seeds 3 [--glitch 1.7]
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np

from ..model.hmm import ALL_EXTRA, decode, extract_evidence, fit
from .common import load_possession_metrics
from .degrade import DegradeParams, calibrated_params, degrade
from .metrica import load_game3

TEMP = 0.3                      # as in src/run.py (chosen on external games, EXP-002)
VARIANTS = [("A team", ()), ("B +out", ("out",)), ("C +crowd", ("out", "crowd")),
            ("D +gapedge", ("out", "crowd", "gapedge"))]
GAIN_MIN = 0.04                 # possession_metrics.SPELL_NOISE_BAND
MACRO_DROP_MAX = 0.02           # EXP-003 guard


def _score(pm, gt, pred) -> Dict[str, float]:
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "dF1": r["per_class"]["D"]["f1"],
            "ad": r["alive_dead_f1"], "segs": r["n_pred_segments"], "gt_segs": r["n_gt_segments"]}


def cross_fit(evidence, gt, runs) -> Dict[str, np.ndarray]:
    preds = {name: np.full(len(gt), 3, dtype=np.int8) for name, _ in VARIANTS}
    for k, test_run in enumerate(runs):
        train = np.zeros(len(gt), dtype=bool)
        for j, (a, b) in enumerate(runs):
            if j != k:
                train[a:b] = True
        params = fit(evidence, gt, train & (gt != 3))
        a, b = test_run
        for name, streams in VARIANTS:
            part = decode(evidence, params, [test_run], temp=TEMP, use_speed=False, streams=streams)
            preds[name][a:b] = part[a:b]
    return preds


def run_setting(label, make_case, seeds) -> Dict[str, List[Dict[str, float]]]:
    pm = load_possession_metrics()
    out: Dict[str, List[Dict[str, float]]] = {name: [] for name, _ in VARIANTS}
    for seed in range(seeds):
        gt, preds = make_case(seed)
        for name, _ in VARIANTS:
            out[name].append(_score(pm, gt, preds[name]))
    print("\n== {} ({} seeds) ==".format(label, seeds))
    print("{:12s} {:>8s} {:>7s} {:>8s} {:>7s} {:>8s} {:>10s}".format(
        "variant", "spellF1", "+-", "macroF1", "D-F1", "alive/dead", "segs(gt)"))
    for name, _ in VARIANTS:
        a = out[name]
        print("{:12s} {:8.4f} {:7.4f} {:8.4f} {:7.4f} {:8.4f} {:5.0f}({:.0f})".format(
            name, np.mean([x["spell"] for x in a]), np.std([x["spell"] for x in a]),
            np.mean([x["macro"] for x in a]), np.mean([x["dF1"] for x in a]),
            np.mean([x["ad"] for x in a]), np.mean([x["segs"] for x in a]), a[0]["gt_segs"]))
    return out


def decide(results: Dict[str, Dict[str, List[Dict[str, float]]]]) -> None:
    """Fixed rule from EXP-003.md: keep a variant over the previous one only if, in EVERY setting, the mean
    spellF1 gain exceeds GAIN_MIN, no seed regresses, D-F1 does not fall and macroF1 falls by <= 0.02."""
    print("\n== decision rule (gain > {:.2f} spellF1 in every setting, no seed regresses, D-F1 not down, "
          "macroF1 drop <= {:.2f}) ==".format(GAIN_MIN, MACRO_DROP_MAX))
    kept = VARIANTS[0][0]
    for name, _ in VARIANTS[1:]:
        ok, notes = True, []
        for setting, res in results.items():
            prev, cur = res[kept], res[name]
            d_spell = np.array([c["spell"] - p["spell"] for p, c in zip(prev, cur)])
            d_macro = np.mean([c["macro"] - p["macro"] for p, c in zip(prev, cur)])
            d_df1 = np.mean([c["dF1"] - p["dF1"] for p, c in zip(prev, cur)])
            cond = d_spell.mean() > GAIN_MIN and d_spell.min() > 0 and d_df1 >= 0 and d_macro >= -MACRO_DROP_MAX
            ok &= bool(cond)
            notes.append("{}: dSpell {:+.3f} (min seed {:+.3f}) dD-F1 {:+.3f} dMacro {:+.3f}".format(
                setting, d_spell.mean(), d_spell.min(), d_df1, d_macro))
        print("{} vs {}: {}".format(name, kept, "KEEP" if ok else "DROP"))
        for n in notes:
            print("    " + n)
        if ok:
            kept = name
    print("selected variant: {}".format(kept))


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--legacy-noise", action="store_true",
                    help="original iid 0.3 m noise + 1.7 glitches/min (first EXP-003 run); default is the "
                         "profile-calibrated degradation")
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args(argv)

    params = DegradeParams(p_ball_glitch_per_min=1.7) if args.legacy_noise else calibrated_params()
    print("degradation: {}".format("legacy iid noise" if args.legacy_noise else "profile-calibrated"))
    clean, gt, info = load_game3()
    runs = info["halves"]
    results = {}

    def metrica_case(seed):
        tr = degrade(clean, seed=1000 + seed, params=params)
        return gt, cross_fit(extract_evidence(tr, streams=ALL_EXTRA), gt, runs)
    results["metrica cross-fit"] = run_setting("Metrica Game 3, 2-fold by half", metrica_case, args.seeds)

    if not args.skip_transfer:
        from .skillcorner import load_match
        sc_clean, sc_gt, sc_info = load_match()
        train_ev = extract_evidence(degrade(clean, seed=999, params=params), streams=ALL_EXTRA)
        fitted = fit(train_ev, gt, gt != 3)

        def transfer_case(seed):
            tr = degrade(sc_clean, seed=2000 + seed, params=params)
            ev = extract_evidence(tr, streams=ALL_EXTRA)
            return sc_gt, {name: decode(ev, fitted, sc_info["halves"], temp=TEMP, use_speed=False, streams=s)
                           for name, s in VARIANTS}
        results["transfer Metrica->SkillCorner"] = run_setting(
            "Transfer: fit Metrica, decode SkillCorner", transfer_case, args.seeds)

    decide(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
