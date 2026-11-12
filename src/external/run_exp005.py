"""EXP-005: does team-shape evidence (compactness asymmetry) improve on variant C? External bench only.

Pre-registered rule in experiments/EXP-005.md: adopt E (C + shape) iff in EVERY setting mean spellF1 gain > 0.04, no
seed regresses, D-F1 does not fall and macroF1 does not fall by more than 0.02.

    python -m src.external.run_exp005 --seeds 3
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from ..model.hmm import decode, extract_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params, degrade
from .metrica import load_game3
from .run_exp003 import TEMP, _score

BASE = ("out", "crowd")                     # variant C
VARIANTS: Dict[str, tuple] = {}             # filled in main(): "C" and "cand" (= C + candidate stream)
ALL: tuple = ()
GAIN_MIN, MACRO_DROP_MAX = 0.04, 0.02


def _report(label, res, dshare, gt_d):
    print("\n== {} | true D share {:.3f} ==".format(label, gt_d))
    print("{:10s} {:>8s} {:>7s} {:>8s} {:>7s} {:>9s} {:>10s}".format(
        "variant", "spellF1", "+-", "macroF1", "D-F1", "pred D %", "segs(gt)"))
    for v, a in res.items():
        print("{:10s} {:8.4f} {:7.4f} {:8.4f} {:7.4f} {:9.3f} {:5.0f}({:.0f})".format(
            v, np.mean([x["spell"] for x in a]), np.std([x["spell"] for x in a]), np.mean([x["macro"] for x in a]),
            np.mean([x["dF1"] for x in a]), np.mean(dshare[v]), np.mean([x["segs"] for x in a]), a[0]["gt_segs"]))


def run_setting(label, case, seeds, results):
    pm = load_possession_metrics()
    res = {v: [] for v in VARIANTS}
    dshare = {v: [] for v in VARIANTS}
    gt_d = 0.0
    for seed in range(seeds):
        gt, preds = case(seed)
        graded = gt != 3
        gt_d = float((gt[graded] == 2).mean())
        for v in VARIANTS:
            res[v].append(_score(pm, gt, preds[v]))
            dshare[v].append(float((preds[v][graded] == 2).mean()))
    _report(label, res, dshare, gt_d)
    results[label] = res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stream", default="shape", help="candidate stream added to C: 'shape' (EXP-005) or 'shape_ha' (EXP-006)")
    args = ap.parse_args(argv)

    global VARIANTS, ALL
    VARIANTS = {"C": BASE, "cand": BASE + (args.stream,)}
    ALL = BASE + (args.stream,)
    print("candidate stream: {}".format(args.stream))

    from .skillcorner import load_match
    params = calibrated_params()
    m_clean, m_gt, m_info = load_game3()
    s_clean, s_gt, s_info = load_match()
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    def metrica_cross(seed):
        tr = degrade(m_clean, seed=1000 + seed, params=params)
        ev = extract_evidence(tr, streams=ALL)
        runs = m_info["halves"]
        preds = {v: np.full(len(m_gt), 3, dtype=np.int8) for v in VARIANTS}
        for k, test_run in enumerate(runs):
            train = np.zeros(len(m_gt), dtype=bool)
            for j, (a, b) in enumerate(runs):
                if j != k:
                    train[a:b] = True
            p = fit(ev, m_gt, train & (m_gt != 3))
            for v, streams in VARIANTS.items():
                part = decode(ev, p, [test_run], temp=TEMP, use_speed=False, streams=streams)
                a, b = test_run
                preds[v][a:b] = part[a:b]
        return m_gt, preds

    def transfer(train_clean, train_gt, test_clean, test_gt, test_runs):
        ev0 = extract_evidence(degrade(train_clean, seed=999, params=params), streams=ALL)
        fitted = fit(ev0, train_gt, train_gt != 3)

        def case(seed):
            ev = extract_evidence(degrade(test_clean, seed=2000 + seed, params=params), streams=ALL)
            return test_gt, {v: decode(ev, fitted, test_runs, temp=TEMP, use_speed=False, streams=s)
                             for v, s in VARIANTS.items()}
        return case

    run_setting("Metrica 2-fold by half", metrica_cross, args.seeds, results)
    run_setting("Transfer Metrica -> SkillCorner", transfer(m_clean, m_gt, s_clean, s_gt, s_info["halves"]),
                args.seeds, results)
    run_setting("Transfer SkillCorner -> Metrica", transfer(s_clean, s_gt, m_clean, m_gt, m_info["halves"]),
                args.seeds, results)

    print("\n== decision rule: E adopted iff in EVERY setting dSpell > {:.2f}, no seed regresses, D-F1 not down, "
          "macroF1 drop <= {:.2f} ==".format(GAIN_MIN, MACRO_DROP_MAX))
    ok_all = True
    for label, res in results.items():
        c, e = res["C"], res["cand"]
        d_sp = np.array([y["spell"] - x["spell"] for x, y in zip(c, e)])
        d_df = np.mean([y["dF1"] - x["dF1"] for x, y in zip(c, e)])
        d_ma = np.mean([y["macro"] - x["macro"] for x, y in zip(c, e)])
        ok = d_sp.mean() > GAIN_MIN and d_sp.min() > 0 and d_df >= 0 and d_ma >= -MACRO_DROP_MAX
        ok_all &= bool(ok)
        print("  {}: dSpell {:+.3f} (min seed {:+.3f}) dD-F1 {:+.3f} dMacro {:+.3f} -> {}".format(
            label, d_sp.mean(), d_sp.min(), d_df, d_ma, "pass" if ok else "fail"))
    print("decision: {}".format("ADOPT C + {}".format(args.stream) if ok_all else "KEEP C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
