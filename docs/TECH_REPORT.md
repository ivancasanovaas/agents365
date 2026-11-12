# Technical report: possession labelling with a hidden Markov model and discriminative emissions

Team: **agents365** (members: Pau Marfil, Ivan Casanovas, Xavi Hernández, Martín). Repository:
https://github.com/ivancasanovaas/agents365. Match: BAR-ATH, tactical timeline (199,319 frames).
Status: the model is selected and validated on **external games only**; no label of the provided match was used for training,
selection or tuning (section 6).

## 1. Approach and why

spellF1 scores possessions, not frames, so the unit that matters is the possession interval. The organiser baseline (nearest
player to the ball, majority vote) decides frame by frame and fragments possessions. We label the match with a **three-state
hidden Markov model** (H, A, D) decoded offline with **Viterbi** inside each graded half. A possession is therefore a run of a
sticky state, and gaps in the tracking are bridged by the state, not by guessing.

**Emissions.** For every frame a **small neural network (one hidden layer of 32 ReLU units, softmax output, numpy only)** gives
P(state | x) from 32 collective features computed from the tracking only. A linear softmax with the same features is the
reference model. The HMM uses the scaled likelihood P(state | x) / P(state) with a temperature of 0.3 (consecutive frames
are strongly correlated, so the evidence is down-weighted). The transition matrix and the initial prior come from counting on the
same training games. Features (all orientation-free, none depends on the anonymous track-slot identities):
- ball visibility and, per team, the distance from the ball to the nearest player, to the 3 nearest, to the team centroid, and the
  number of players within 5 m and 10 m of the ball (the ball is missing on 41% of graded frames, so these carry a validity flag);
- ball speed over a 1 s window (12 frames each side), ball-outside-the-pitch flag with the length bucket of the outside run
  (FIFA Law 9);
- collective stillness: median player speed over the same 1 s window, in coarse buckets (a dead ball shows up as players that stop);
- team shape: compactness (mean pairwise distance) and spread of each team, their differences, and team speeds. The team that has
  the ball is the more spread one and the defender compacts; this signal is strongly informative but only useful when modelled
  jointly with the others (a plain product of per-signal likelihoods made it harmful, section 4).
Home/away symmetry is enforced by also training on the team-swapped copy of every frame. The 2,934 graded frames before the
second-half kickoff are forced to `D` (published rule in FAQ.md); everything outside the graded halves is `X`.

## 2. Assets used and dropped

Used: the tactical ATD tracking and its metadata (team of each slot), through the organiser's `games.py` / `starter/baseline.py`
helpers. **Smart Tagging is not used by the model**: we only use it afterwards as an independent consistency check (our labels
agree with its `Team` tag 78% of the time on frames where we say H or A, up from 64% with the previous model; note that the
`Team` of the `DEFENDING...` codes is the defending side). Homography, both videos and the broadcast tracking are not used.
No private or in-house model is used.

Tried and dropped (each one pre-registered with a fixed decision rule, see `experiments/`): ball-speed stream in the naive-Bayes
model; ball-outside + stillness streams combined with a gap-edge stream (EXP-003); per-game quantile bucketing of stillness
(EXP-004); three ways of adding team shape as a naive-Bayes stream (EXP-005/006/007); duration handling by H<->A damping and a
minimum-duration filter (EXP-009); softmax temperatures 0.1 (EXP-010) and 0.5/0.7/1.0 (EXP-011); 19 temporal-context features
(EXP-012). The one-hidden-layer MLP (EXP-013: +0.027 spellF1 over the linear softmax, positive in 7 of 7 development games, below
the pre-registered 0.04 margin) was **adopted only after** a pre-registered confirmation on four untouched games (EXP-014).

## 3. Training and validation data (other games only)

| Dataset | Source | Labeller | Use |
|---|---|---|---|
| Metrica Sample Games 1, 2, 3 (145,006 / 141,156 / 143,761 frames at 25 fps) | github.com/metrica-sports/sample-data (no formal licence; responsible use and credit requested) | the provider (Metrica) produced the tracking and the human-annotated events; **we label nothing**: `src/external/metrica.py`, `metrica_csv.py` and `labels.py` convert the events to H/A/D by a fixed algorithm | training and validation |
| SkillCorner open data, 4 A-League matches: 1874553, 1886347, 1899585, 1925299 (150,752 / 147,652 / 151,325 / 153,252 frames after resampling from 10 to 25 fps) | github.com/SkillCorner/opendata (MIT) | the provider (SkillCorner) produced the tracking and the phases of play; we convert the phases to H/A/D with `src/external/skillcorner.py` | training and validation |

All files are pinned to a source commit and verified by sha256 (`scripts/fetch_external.py`). Total 1,032,904 frames.
In addition, **four further SkillCorner matches (1927964, 1953632, 1959846, 1986691) are an untouched hold-out set**: they are
excluded from `registry.available()`, never used for training, tuning or selection (a test checks that no model file mentions
them) and were scored once, in EXP-014. Labels follow the competition definition: control events (pass, carry,
shot, recovery, set piece) set the team; challenges and lost balls do not change it ("control, not touch"); dead ball runs from a
ball-out, foul or goal to the next restart. The external D comes from events/phases while the competition's comes from corrected
tracking, so absolute numbers differ; the mechanism is what transfers.

**Degrader.** External tracking is cleaner than the competition's, so it is corrupted by a seeded degrader (ball missing 41%,
~20.5 players present, same-team identity swaps, slot anonymisation, correlated position noise, spurious ball jumps, long ball
gaps). Its noise, glitch rate and gap length are **calibrated against aggregate statistics of the competition tracking** (ball-step
speed quantiles, share of steps above 40 m/s, median-player speed, length of ball-missing runs), with no labels; the recipe is
`scripts/calibrate_degrade.py` and `PROVENANCE.md`. The first version of the bench was far too noisy and had ball gaps of ~2 s;
after calibration the realised gaps have a median of ~7 s and a p90 of ~37 s (the competition's are 10.7 s and 34 s), so the bench
is still somewhat milder than the real tracking in the median. Both corrections were made after the first experiments;
EXP-003/004 and the team-shape experiments were re-run on the corrected bench (EXP-008) and kept their decisions (apart from the
disclosed judgement call below); EXP-009 to EXP-013 were run on it. The very early comparison of EXP-002 (ball-speed stream,
temperature) was not re-run, but the temperature was re-checked in EXP-010/011.

## 4. Validation protocol and results

Every decision from EXP-003 on was taken on external games with a decision rule written in `experiments/EXP-0xx.md` **before**
running. EXP-003 to EXP-009 used 2-game settings (Metrica cross-fitted by half; Metrica -> SkillCorner; from EXP-004 also
SkillCorner -> Metrica), requiring a spellF1 gain above 0.04 (the scorer's noise band) in every setting. From EXP-010 on, with
seven games, the protocol is **leave-one-game-out** (train on six, test on the seventh, three degradation seeds): a variant is
adopted only if the mean spellF1 gain exceeds 0.04, is positive in at least 5 of the 7 games, and loses no more than 0.02 of
D-F1 or macroF1. All seven development games were used repeatedly in these decisions, so those numbers are optimistic; the
four hold-out games (EXP-014) give an estimate unaffected by model selection, except for one decision (MLP versus softmax).

| Mean spellF1 / macroF1 / D-F1 | 7 development games, leave-one-game-out | **4 untouched hold-out games** |
|---|---|---|
| team-only naive Bayes (A) | 0.379 / 0.576 / 0.436 | 0.385 / 0.554 / 0.363 |
| + ball-outside and stillness streams (C) | 0.434 / 0.746 / 0.782 | 0.429 / 0.783 / 0.830 |
| linear softmax emissions | 0.516 / 0.810 / 0.834 | 0.541 / 0.856 / 0.899 |
| **MLP emissions (production)** | 0.543 / 0.822 / 0.844 | **0.564 / 0.870 / 0.910** |

The softmax beats C in 7 of 7 development games (+0.082 spellF1) and by +0.112 on the hold-out. The MLP beats the softmax in 7 of 7
development games (+0.027) and in 3 of 4 hold-out games (+0.023); all hold-out games are SkillCorner A-League matches. The organiser baseline (smooth50) scores 0.243 on the full match. On the
provided match we only inspect **aggregates of our own output** (no score is available to us): dead-ball share 44.5% of graded
frames (the external games range from 30% to 51%), 117 home and 114 away possessions with 103 direct H<->A switches, and 77%
agreement with the tagged `Team` in the Smart Tagging (a consistency check, never a selection criterion; the softmax gave 78%, so
this check does not by itself confirm the external gain). The output is bit-for-bit deterministic (same sha256 on repeated runs).

**Fresh cross-provider test (EXP-016) and a negative result (EXP-017).** After the decisions above we downloaded seven Bundesliga
matches from the Sportec Open DFL Dataset (CC BY 4.0), whose tracking comes with native per-frame ball status and owning team, the
same definition as the competition's ground truth (dead ball, otherwise the team the tracking names as owning it). Scoring the
production models (trained on none of them) gave the same ranking A < C < softmax < MLP: MLP D-F1 0.855, alive/dead F1 0.873,
team accuracy 0.708, spellF1 0.359. spellF1 is not comparable with the other tables because DFL labels flip possession about twice
as often (median spell 3 s) as the competition's public example. The models over-predict dead ball by 3-9 points there. EXP-017
asked whether adding these native-label games to the emission training is worth it: on the DFL folds D-F1 rose to 0.935 and team
accuracy to 0.792, but on the original development folds spellF1 fell from 0.543 to 0.531 (more than the 0.01 no-harm tolerance
fixed beforehand), so the pre-registered rule says **keep production as is**. We report it as is: native-label training clearly
helps dead-ball and team accuracy, at some cost in possession granularity, and we did not adopt it. A 4-state HMM whose dead-ball
state remembers which team had the ball (EXP-015, an idea from the PathCRF paper) gave identical results and was rejected.

**Deviations from the pre-registered rules, disclosed.** (a) EXP-003: the definition of variant C was amended before any result was
seen (crowd stillness instead of ball stationary, because the aggregate profile showed the latter is rare; assumed bucket edges
instead of edges estimated on external games). (b) EXP-003 on the corrected bench: the literal outcome of the rule was variant D
(gap-edge stream) because each variant was compared with the last kept one, never with C; compared directly D is no better than C
and clearly worse in the transfer (D-F1 -0.20), so **C was kept as a judgement call, not as the rule's output**. In the first run C
was also kept with a minimum per-seed gain of +0.031 under a rule that asked for gains "consistent across seeds". (c) EXP-013
rejected the MLP by the 0.04 margin although it was better in 7 of 7 games; it was then adopted through a separate pre-registered
confirmation on untouched games (EXP-014) whose margin (+0.02, 3 of 4 games) is lower than the general one; the MLP passed with
+0.023 and 3 of 4. The later experiments compare each candidate with the production variant.

## 5. Possession rules we implemented

A possession is a maximal run of H or A with the ball alive; D splits possessions. There are no hand-written rules for loose
balls, challenges, aerial duels or clearances: possession stays with the last team until the evidence for the other team is
sustained (the stickiness of the HMM and the temporal averaging of the features), which follows the "control, not touch"
definition. Restarts are not modelled explicitly beyond the dead-ball cues (ball outside the pitch, collective stillness).

## 6. Compliance statement

- No hand annotation or hand correction of the provided match. No human label about it was used, in any amount.
- The private ground truth was never available to us. The public five-minute ground truth was used only to check that the scorer
  and pipeline run (early baseline sanity check); it was never used to train, tune or choose among variants.
- Every threshold and constant has an origin in `PROVENANCE.md` (FIFA Laws, physics, external open data, aggregate statistics of
  the tracking without labels, or a coarse assumption stated as such). Experiments, including the negative ones, are in
  `experiments/`.
- AI assistants: development was done with **Claude Code, a cloud service (Anthropic)**. It was configured to deny reads of the
  data folder (a hook blocks shell commands that touch it), so no row, position or label of the match was given to it. What the
  team did paste into the conversation, from `src/profile_datasets.py` (the exact generator of those numbers), were **aggregate
  statistics, never per-frame values**: counts and shares of the tracking (ball visibility, number of objects), quantiles of ball
  speed, of ball-missing run lengths and of player speed, ball-outside-the-pitch shares, Smart Tagging counts and durations per
  code, alignment offsets between set-piece tags and physical cues, and aggregates of **our own output** (label shares, run
  durations, H<->A switches, agreement with the tagged `Team`). They were used to calibrate the degrader and to motivate
  hypotheses; **none was used to choose among model variants** (that was done only on external games), and our own output and the
  Smart Tagging were never used for training or selection. Some of these aggregates are also written in `PROVENANCE.md`,
  `scripts/calibrate_degrade.py` and `experiments/`, so they are public in the repository. The terms forbid giving "extracts" of
  the data to a cloud LLM; we consider aggregate statistics not to be extracts, report it here in full, and will follow the
  organisers' ruling. A local open model (Ollama, `qwen3.8:27b`) was installed on the DGX for analysis, but no
  result in this report depends on it.
- The match data is not committed or redistributed; `data/` and `ext_open/` are git-ignored.

## 7. Limitations and what we would do with more time

- External labels come from events/phases, the competition's from corrected tracking: the dead-ball definition and the exact
  moments of the changes of possession differ, so the bench gains are evidence for the mechanism, not a guarantee on the match.
- Seven development games from two providers (plus 4 SkillCorner hold-out and 7 DFL evaluation games, not used for training); the entry cost of a new game is one adapter. More leagues would reduce the risk of a
  shift in team shapes.
- The temperature (0.3), the MLP size and the feature scaling are not tuned beyond the pre-registered comparisons. Longer temporal
  context (EXP-012) and duration handling (EXP-009) gave little; a larger network, an ensemble and a proper explicit-duration
  (semi-Markov) model are untested (see `experiments/`).
- All four hold-out games and most development games are from one provider (SkillCorner, A-League), so a shift to LaLiga team
  shapes is only partly covered by the Metrica games.
- The model has no notion of who restarts a dead ball and no explicit clearance/duel logic.

## 8. Reproduction

```
pip install -r requirements.lock.txt
git config core.hooksPath scripts/hooks       # opt-in guard: blocks data, CSVs and notebooks with outputs at commit time
# optional: the committed models/ already hold the fitted parameters; to refit them from the open external games:
python scripts/fetch_external.py               # pinned download of the 7 external games into ext_open/, verified by sha256
python -m src.fit_params                       # refits models/hmm_params.json, softmax_params.json and mlp_params.json
python -m src.run --data-dir /path/to/data     # writes outputs/submission.csv and validates it
python evaluation/validate_submission.py outputs/submission.csv
python -m unittest discover -s tests           # synthetic data only
```

Python 3.12, numpy and pandas (pinned in `requirements.lock.txt`); the pipeline has no randomness and writes the same file on
every run (verify with `sha256sum` on two runs).
