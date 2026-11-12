"""EXP-009: possession-duration handling on top of variant C (external bench v2 only).

Candidates (experiments/EXP-009.md): H<->A transition scaling (0.5 / 0.25 / 0.1) and a minimum-duration post-filter whose
threshold is the 5% / 10% quantile of true possession durations in the TRAINING game(s). Each candidate is compared
DIRECTLY with C in the same run; the pre-registered rule is applied mechanically at the end.

    python -m src.external.run_exp009 --seeds 3
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from ..model.hmm import decode, extract_evidence, fit, scale_ha
from ..model.postfilter import min_duration_filter, spell_duration_quantile
from .common import FPS, load_possession_metrics
from .degrade import calibrated_params, degrade
from .metrica import load_game3
from .run_exp003 import TEMP, _score

STREAMS = ("out", "crowd")
VARIANTS = {
    "C": (1.0, None),
    "ha 0.5": (0.5, None), "ha 0.25": (0.25, None), "ha 0.1": (0.1, None),
    "filter q05": (1.0, 0.05), "filter q10": (1.0, 0.10),
}
QUANTILES = (0.05, 0.10)
GAIN_MIN, DROP_MAX = 0.04, 0.02


def predict_all(ev, fitted, test_runs, train_gt, train_runs) -> Dict[str, np.ndarray]:
    thresholds = {q: spell_duration_quantile(train_gt, train_runs, q) for q in QUANTILES}
    out = {}
    for name, (scale, q) in VARIANTS.items():
        params = scale_ha(fitted, scale) if scale != 1.0 else fitted
        codes = decode(ev, params, test_runs, temp=TEMP, use_speed=False, streams=STREAMS)
        out[name] = min_duration_filter(codes, thresholds[q], test_runs) if q else codes
    return out, thresholds


def run_setting(label, case, seeds, results):
    pm = load_possession_metrics()
    res = {v: [] for v in VARIANTS}
    thr = None
    for seed in range(seeds):
        gt, preds, thr = case(seed)
        for v in VARIANTS:
            res[v].append(_score(pm, gt, preds[v]))
    print("\n== {} | thresholds q05/q10 = {:.1f}/{:.1f} s ==".format(
        label, thr[0.05] / FPS, thr[0.10] / FPS))
    print("{:12s} {:>8s} {:>7s} {:>8s} {:>7s} {:>10s}".format("variant", "spellF1", "+-", "macroF1", "D-F1", "segs(gt)"))
    for v, a in res.items():
        print("{:12s} {:8.4f} {:7.4f} {:8.4f} {:7.4f} {:5.0f}({:.0f})".format(
            v, np.mean([x["spell"] for x in a]), np.std([x["spell"] for x in a]), np.mean([x["macro"] for x in a]),
            np.mean([x["dF1"] for x in a]), np.mean([x["segs"] for x in a]), a[0]["gt_segs"]))
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
        ev = extract_evidence(tr, streams=STREAMS)
        runs = m_info["halves"]
        preds = {v: np.full(len(m_gt), 3, dtype=np.int8) for v in VARIANTS}
        thr_all = None
        for k, test_run in enumerate(runs):
            train_runs = [r for j, r in enumerate(runs) if j != k]
            train = np.zeros(len(m_gt), dtype=bool)
            for a, b in train_runs:
                train[a:b] = True
            fitted = fit(ev, m_gt, train & (m_gt != 3))
            part, thr_all = predict_all(ev, fitted, [test_run], m_gt, train_runs)
            a, b = test_run
            for v in VARIANTS:
                preds[v][a:b] = part[v][a:b]
        return m_gt, preds, thr_all

    def transfer(train_clean, train_gt, train_runs, test_clean, test_gt, test_runs):
        ev0 = extract_evidence(degrade(train_clean, seed=999, params=params), streams=STREAMS)
        fitted = fit(ev0, train_gt, train_gt != 3)

        def case(seed):
            ev = extract_evidence(degrade(test_clean, seed=2000 + seed, params=params), streams=STREAMS)
            preds, thr = predict_all(ev, fitted, test_runs, train_gt, train_runs)
            return test_gt, preds, thr
        return case

    run_setting("Metrica 2-fold by half", metrica_cross, args.seeds, results)
    run_setting("Transfer Metrica -> SkillCorner",
                transfer(m_clean, m_gt, m_info["halves"], s_clean, s_gt, s_info["halves"]), args.seeds, results)
    run_setting("Transfer SkillCorner -> Metrica",
                transfer(s_clean, s_gt, s_info["halves"], m_clean, m_gt, m_info["halves"]), args.seeds, results)

    print("\n== decision rule: candidate adopted iff in EVERY setting dSpell > {:.2f} vs C, no seed regresses, "
          "D-F1 drop <= {:.2f}, macroF1 drop <= {:.2f} ==".format(GAIN_MIN, DROP_MAX, DROP_MAX))
    passing = []
    for v in VARIANTS:
        if v == "C":
            continue
        ok, gains, notes = True, [], []
        for label, res in results.items():
            c, x = res["C"], res[v]
            d_sp = np.array([b["spell"] - a["spell"] for a, b in zip(c, x)])
            d_df = np.mean([b["dF1"] - a["dF1"] for a, b in zip(c, x)])
            d_ma = np.mean([b["macro"] - a["macro"] for a, b in zip(c, x)])
            cond = d_sp.mean() > GAIN_MIN and d_sp.min() > 0 and d_df >= -DROP_MAX and d_ma >= -DROP_MAX
            ok &= bool(cond)
            gains.append(d_sp.mean())
            notes.append("{}: dSpell {:+.3f} (min seed {:+.3f}) dD-F1 {:+.3f} dMacro {:+.3f}".format(
                label, d_sp.mean(), d_sp.min(), d_df, d_ma))
        print("{}: {}".format(v, "PASS" if ok else "fail"))
        for n in notes:
            print("    " + n)
        if ok:
            passing.append((float(np.mean(gains)), v))
    if passing:
        best = max(passing)[1]
        print("decision: ADOPT {} (highest mean gain among passing: {})".format(best, [p[1] for p in passing]))
    else:
        print("decision: KEEP C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
