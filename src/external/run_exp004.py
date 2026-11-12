"""EXP-004: is per-game quantile bucketing of the crowd-stillness stream non-inferior to fixed absolute edges?

Motivation (aggregate profile of the competition tracking vs the calibrated bench): the competition median player is
slower than Metrica's (crowd-speed bucket shares 0.066/0.129/0.211/0.314/0.280 vs 0.040/0.101/0.174/0.283/0.401), so
fixed absolute edges may read ordinary play as stillness. Per-game quantile edges remove overall tempo differences
without labels. The bench can only check that nothing is lost, so the pre-registered rule is NON-INFERIORITY.

Settings: Metrica 2-fold by half; transfer Metrica -> SkillCorner; transfer SkillCorner -> Metrica.
Rule (fixed in experiments/EXP-004.md before running): adopt the quantile edges iff in EVERY setting the mean
spellF1 and macroF1 are each no more than 0.02 below the absolute-edge variant.

    python -m src.external.run_exp004 --seeds 3
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from ..model import features as F
from ..model.hmm import EXTRA_NSYM, decode, extract_evidence, fit
from .common import load_possession_metrics
from .degrade import calibrated_params, degrade
from .metrica import load_game3
from .run_exp003 import TEMP, _score

STREAMS = ("out", "crowd")
VARIANTS = ("C absolute", "C quantile")
NONINF = 0.02


def _evidence(tr, runs, variant):
    edges = F.crowd_edges_for_game(tr.players, runs) if variant == "C quantile" else None
    return extract_evidence(tr, streams=STREAMS, crowd_edges=edges)


def _support_report(ev, gt, name):
    """Aggregate: training-frame counts per 'out' symbol and state, to see which weights are learned vs smoothed."""
    sym = ev.extra["out"]
    rows = []
    for k in range(EXTRA_NSYM["out"]):
        sel = sym == k
        rows.append("sym{}: D={} alive={}".format(k, int(((gt == 2) & sel).sum()), int((((gt == 0) | (gt == 1)) & sel).sum())))
    print("  [{}] 'out' stream support in training frames: {}".format(name, " | ".join(rows)))


def run_setting(label, case, seeds, results):
    pm = load_possession_metrics()
    res = {v: [] for v in VARIANTS}
    dshare = {v: [] for v in VARIANTS}
    gt_d = None
    for seed in range(seeds):
        gt, preds = case(seed)
        gt_d = float((gt == 2).sum() / max((gt != 3).sum(), 1))
        for v in VARIANTS:
            res[v].append(_score(pm, gt, preds[v]))
            graded = gt != 3
            dshare[v].append(float((preds[v][graded] == 2).mean()))
    print("\n== {} ({} seeds) | true D share of graded frames {:.3f} ==".format(label, seeds, gt_d))
    print("{:12s} {:>8s} {:>7s} {:>8s} {:>7s} {:>10s}".format("variant", "spellF1", "+-", "macroF1", "D-F1", "pred D %"))
    for v in VARIANTS:
        a = res[v]
        print("{:12s} {:8.4f} {:7.4f} {:8.4f} {:7.4f} {:10.3f}".format(
            v, np.mean([x["spell"] for x in a]), np.std([x["spell"] for x in a]),
            np.mean([x["macro"] for x in a]), np.mean([x["dF1"] for x in a]), np.mean(dshare[v])))
    results[label] = res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    from .skillcorner import load_match
    params = calibrated_params()
    m_clean, m_gt, m_info = load_game3()
    s_clean, s_gt, s_info = load_match()
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    def metrica_cross(seed):
        tr = degrade(m_clean, seed=1000 + seed, params=params)
        runs = m_info["halves"]
        preds = {v: np.full(len(m_gt), 3, dtype=np.int8) for v in VARIANTS}
        for v in VARIANTS:
            ev = _evidence(tr, runs, v)
            for k, test_run in enumerate(runs):
                train = np.zeros(len(m_gt), dtype=bool)
                for j, (a, b) in enumerate(runs):
                    if j != k:
                        train[a:b] = True
                p = fit(ev, m_gt, train & (m_gt != 3))
                part = decode(ev, p, [test_run], temp=TEMP, use_speed=False, streams=STREAMS)
                a, b = test_run
                preds[v][a:b] = part[a:b]
        return m_gt, preds

    def transfer(train_clean, train_gt, train_info, test_clean, test_gt, test_info, tag):
        fitted = {}
        for v in VARIANTS:
            tr0 = degrade(train_clean, seed=999, params=params)
            ev0 = _evidence(tr0, train_info["halves"], v)
            if v == VARIANTS[0]:
                _support_report(ev0, train_gt, tag)
            fitted[v] = fit(ev0, train_gt, train_gt != 3)

        def case(seed):
            tr = degrade(test_clean, seed=2000 + seed, params=params)
            preds = {v: decode(_evidence(tr, test_info["halves"], v), fitted[v], test_info["halves"],
                               temp=TEMP, use_speed=False, streams=STREAMS) for v in VARIANTS}
            return test_gt, preds
        return case

    run_setting("Metrica 2-fold by half", metrica_cross, args.seeds, results)
    run_setting("Transfer Metrica -> SkillCorner",
                transfer(m_clean, m_gt, m_info, s_clean, s_gt, s_info, "train=Metrica"), args.seeds, results)
    run_setting("Transfer SkillCorner -> Metrica",
                transfer(s_clean, s_gt, s_info, m_clean, m_gt, m_info, "train=SkillCorner"), args.seeds, results)

    print("\n== decision rule: quantile edges adopted iff spellF1 and macroF1 are each within {:.2f} of absolute "
          "edges in EVERY setting ==".format(NONINF))
    ok_all = True
    for label, res in results.items():
        d_sp = np.mean([q["spell"] - a["spell"] for a, q in zip(res[VARIANTS[0]], res[VARIANTS[1]])])
        d_ma = np.mean([q["macro"] - a["macro"] for a, q in zip(res[VARIANTS[0]], res[VARIANTS[1]])])
        ok = d_sp >= -NONINF and d_ma >= -NONINF
        ok_all &= bool(ok)
        print("  {}: dSpell {:+.3f} dMacro {:+.3f} -> {}".format(label, d_sp, d_ma, "non-inferior" if ok else "INFERIOR"))
    print("decision: {}".format("ADOPT quantile edges" if ok_all else "KEEP absolute edges"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
