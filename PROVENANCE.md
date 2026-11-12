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
| Local LLM (analysis assistant only, never labels frames) | Ollama 0.40.3, model `qwen3.8:27b`, ID aaee06c39dcf (17 GB) | Open model pulled via Ollama on the DGX Spark; own instance bound to 127.0.0.1:11435, no cloud, no API keys. Used to propose features/code from aggregate profiles; every threshold it suggests still needs its own row here | DGX Spark (not in repo) | 2026-11-11 |
| Reporting levels in the profile: 10/20/30/40 m/s, 0.5 m/s (ball and crowd stillness), 1 s min run, 2 m near-line | descriptive values | NOT model constants: used only to describe the data in outputs/profile.md; none feeds the pipeline. If promoted to a feature threshold they need their own origin row | src/profile_datasets.py | 2026-11-11 |
| OUT_RUN_EDGES_S / CROWD_EDGES / GAP_EDGE_EDGES | 0.5,2,6 s / 0.4,0.8,1.2,1.8 m/s / 0,2,6,15 m | ASSUMPTION (coarse buckets, not tuned); the learned weights come from external-game counts | src/model/hmm.py | 2026-11-11 |
| CROWD_LAG | 12 frames each side (~1 s) | ASSUMPTION: window long enough to beat position noise, short enough to resolve stoppages | src/model/features.py | 2026-11-11 |
| Ball outside pitch rectangle (cue) | 105.3 x 68.0 m | FIFA Law 9 (ball out of play when it crosses a line) + published pitch size (games.py) | src/model/features.py | 2026-11-11 |
| Degrader calibration: noise_corr 0.999, p_ball_glitch_per_min 0.7 | AR(1) rho, glitches/min | Unsupervised match to AGGREGATE statistics of the competition tracking (ball step 5.1/14.3/24.6 m/s, 0.23% >40 m/s, crowd step 0.56/1.34/3.07 m/s); no labels; scripts/calibrate_degrade.py. glitch_max_frames 3 and margin 10 m are ASSUMPTIONS | src/external/degrade.py::calibrated_params | 2026-11-11 |
| STREAMS = out + crowd (variant C) | selected by fixed rule | EXP-003 pre-registered decision rule on external games only (see experiments/EXP-003.md) | src/run.py, src/fit_params.py | 2026-11-11 |
| HMM parameters pooled on Metrica Game 3 + SkillCorner 1874553 | counts on pooled external games | EXTERNAL labelled games only; pooled because the "ball outside > 2 s" cue has no training frames in Metrica alone (EXP-004 support report). Not validated leave-one-game-out (only two games) | src/fit_params.py, models/hmm_params.json | 2026-11-12 |
| Crowd stream edges: absolute 0.4/0.8/1.2/1.8 m/s (kept) | ASSUMPTION coarse | Per-game quantile edges tested in EXP-004 and rejected by the pre-registered non-inferiority rule | src/model/hmm.py | 2026-11-12 |
| Degrader ball gaps: mean_gap_frames 400 (16 s) | bench v2 | Unsupervised match to AGGREGATE statistics of the competition tracking: 150 ball-missing runs, median 10.7 s, p90 33.8 s, mean ~17 s (outputs/profile.md); geometric run with mean 400 frames has median 11.1 s, p90 36.8 s. No labels; chosen BEFORE running EXP-008. Bench v1 had an ASSUMED 50 frames | src/external/degrade.py::calibrated_params | 2026-11-12 |
| SHAPE_EDGES -8,-3,0,3,8 m | ASSUMPTION coarse | Team-shape streams (EXP-005/006/007) were rejected by their pre-registered rules and are not used in production; kept only for reproducibility of the experiments | src/model/hmm.py | 2026-11-12 |
| NOISE_BAND | 0.04 | possession_metrics.SPELL_NOISE_BAND | run_exp001.py | 2026-11 |
