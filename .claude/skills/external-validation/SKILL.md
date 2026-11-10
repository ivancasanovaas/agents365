---
name: external-validation
description: Use to evaluate possession methods on OTHER games (open datasets) instead of the competition match. Covers dataset choice, deriving H/A/D labels from events, degrading clean tracking to mimic the competition data, and reusing evaluation/possession_metrics.py.
---

# external-validation

Purpose: pick variants and constants on external games, never on the competition match or its public GT.

## Rules
- Only open datasets of OTHER games. Record source, licence, size and match ids in PROVENANCE.md and the tech report.
- Labels are derived by ALGORITHM from the external events (human-produced data about other games is allowed).
- Degrader parameters may come only from numbers PUBLISHED in DATA.md/CHALLENGE.md
  (e.g. ball missing on 41% of graded frames, about 20.5 objects per frame, 46 slots, 25 fps,
  pitch 105.3 x 68.0 m). Do NOT calibrate on statistics computed from the competition data without
  asking the organisers: aggregates of the match may count as extracts under the T&C.
- Everything seeded and deterministic; runs on Linux and Windows (pathlib).

## Pipeline (to implement in src/external/)
1. `load`: kloppy loader per dataset -> normalised internal format (frames x players/ball, 0-1 coords,
   105.3 x 68.0 m, 25 fps via resampling).
2. `labels`: derive per-frame H/A/D from events (rules documented in PROVENANCE.md; the competition GT
   is "D where the corrected tracking marks the ball dead, else the team the tracking names").
3. `degrade(seed, p_ball_missing, gap_model, p_id_switch, drop_rate, pos_noise)`: pure function of the
   clean tracking; anonymises players into slots, injects NaN ball gaps and identity swaps.
4. `score`: build `gt_codes` (use `X` outside play, so `valid_runs` finds the graded spans) and
   `pred_codes` with `possession_metrics.label_codes`, then call
   `possession_metrics.evaluate(gt_codes, pred_codes, bootstrap_n=1000)`. Never copy metric code.
5. Report mean and spread across games and seeds; differences under ~0.04 spellF1 are noise.

## Output
Aggregate metrics only. Never paste competition data.
