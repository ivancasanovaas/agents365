# PROVENANCE: origin of every threshold and constant

Rule: no number enters the pipeline without a row here. Allowed origins: FIFA Laws of the Game,
physics/kinematics, open datasets of OTHER games (cite), or documented unsupervised estimation on the
provided game with no human labels. Not allowed: eyeballing the video, tuning on the public GT.

| Constant | Value | Origin (type + citation) | Where used | Date / author |
|---|---|---|---|---|
| (example) free-kick wall distance | 9.15 m | FIFA Law 13 | dead-ball detector | |
| p_ball_missing | 0.41 | Published: DATA.md / CHALLENGE.md (41% of graded frames) | src/external/degrade.py | 2026-11 |
| drop_rate | 0.068 | Derived from published ~20.5 of 22 objects per frame (DATA.md) | degrade.py | 2026-11 |
| pitch, fps | 105.3x68.0 m, 25 | games.py / DATA.md | src/external/common.py | 2026-11 |
| smoothing window | 50 frames | Organiser baseline_smooth50 (reproduction only, not tuned) | EXP-001 | 2026-11 |
| mean_gap_frames | 50 | ASSUMPTION (unpublished); sweep it, do not tune on competition data | degrade.py | 2026-11 |
| p_switch_per_min, switch_radius_m, pos_noise_m, mean_drop_frames | 0.5, 5 m, 0.3 m, 40 | ASSUMPTIONS (unpublished); sweep parameters | degrade.py | 2026-11 |
| External dataset | Metrica Sample Game 3 (EPTS FIFA tracking + JSON events, ~55 MB) | github.com/metrica-sports/sample-data; no formal licence, asks for responsible use and credit | ext_open/metrica_game3 (gitignored) | 2026-11 |
| External dataset | SkillCorner open data, match 1874553 (A-League 24/25; tracking 10 fps via Git LFS ~91 MB, events, phases, ~96 MB total) | github.com/SkillCorner/opendata, MIT licence, credit SkillCorner; labels derived from phases_of_play | ext_open/skillcorner_1874553 (gitignored) | 2026-11 |
| HMM transition, emission tables | estimated | Supervised counts on EXTERNAL labels (Metrica Game 3, events-derived), cross-fitted by half; never on the competition match or its public GT | src/model/hmm.py | 2026-11 |
| DIST_EDGES / SPEED_EDGES / LAG | 1,2,4,8 m / 0.5,2,6,12 m/s / 5 frames | ASSUMPTION (coarse buckets, not tuned) | hmm.py | 2026-11 |
| HMM temp | 0.3 (candidate) | Chosen on a sweep over EXTERNAL Game 3 only (provisional, one game) | hmm.py | 2026-11 |
| NOISE_BAND | 0.04 | possession_metrics.SPELL_NOISE_BAND | run_exp001.py | 2026-11 |
