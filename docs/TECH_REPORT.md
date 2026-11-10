# Technical report: possession labelling with an HMM over noisy tracking

Team: `<TEAM_NAME>` (members: `<NAMES>`). Repository: `<PUBLIC_REPO_URL>`. Draft status: the pipeline
has been validated on external games only; sections marked TODO must be completed after the first
run on the competition data.

## 1. Approach and why

The task is sequence labelling scored by possession spells (spellF1), so the unit that matters is the
possession interval, not the frame. The organiser baseline (nearest tracked player to the ball + majority
vote) is built from per-frame decisions and over-fragments: on the public five minutes its raw variant
reports 131 possessions against 17 real ones.

We model the match as a **three-state hidden Markov model** (H = home possession, A = away possession,
D = dead ball) decoded offline with **Viterbi** inside each graded half. Offline decoding uses both past
and future frames. States are sticky, so isolated flicker is absorbed and possessions come out as
intervals.

Per-frame evidence (naive Bayes, tempered by a factor `temp = 0.3` because consecutive frames are strongly
correlated):

- team of the nearest tracked player to the ball, bucketed by distance (< 1, 1-2, 2-4, 4-8, > 8 m);
- when the ball is not visible (41% of graded frames) the frame adds **no** evidence; the transition prior
  carries the previous possession across the gap. We never turn a missing ball into `D`.

Ball speed was implemented as a second evidence stream but is **disabled**: it helped within one game and
did not help when transferring to another (section 3).

Frames before the second-half kickoff (graded, all `D` per FAQ.md) are forced to `D` using the published
kickoff frame; everything outside the graded halves is `X`.

## 2. Assets used and dropped

Used: the tactical ATD tracking (`*_atd.txt`) and its metadata (team of each track slot), via the organiser's
`games.py` and `starter/baseline.py` helpers.

Not used: Smart Tagging, homography, both videos, the broadcast ATD. Tried and dropped: ball-speed evidence;
labelling ball-less windows `X` instead of `D` in the baseline (EXP-001, no spellF1 gain, large macroF1 loss).
No private or in-house model is used.

## 3. Training and validation data (other games only)

No label from the provided match is used anywhere: no hand labels, no public-GT training or tuning.

| Dataset | Source | Size | Labeller | Use |
|---|---|---|---|---|
| Metrica Sports Sample Game 3 | github.com/metrica-sports/sample-data (no formal licence; asks for responsible use and credit) | 143,761 frames at 25 fps, 3,620 events | none by us: labels derived **algorithmically** from the provider's event data (human-produced by the provider) | HMM parameter estimation; development benchmark |
| SkillCorner open data, match 1874553 (A-League 2024/25) | github.com/SkillCorner/opendata (MIT, credit SkillCorner) | 60,301 frames at 10 fps, resampled to 25 fps | none by us: labels derived algorithmically from the provider's phases of play | out-of-game transfer test |

The HMM parameters (transition matrix, emission tables) are **estimated by counting** on Metrica labels and
stored in `models/hmm_params.json`; `python -m src.fit_params` regenerates them. Constants and their
origins are listed in `PROVENANCE.md`.

**Degrader.** Because the external tracking is cleaner than the competition's, we corrupt it with a seeded,
parametrised degrader (`src/external/degrade.py`): total ball-missing rate raised to the published 41%,
players dropping out (~20.5 of 22 present on average, published), same-team identity swaps, position noise
and anonymised slots. Gap length, swap rate and noise are **assumptions** (not published) and are recorded as such.

**Results (external games, spellF1 / macroF1, 3 seeds, no competition data):**

| Setting | smooth50 baseline | HMM (team evidence, temp 0.3) |
|---|---|---|
| Metrica, 2-fold cross-fit by half | 0.263 / 0.700 | 0.407 / 0.589 |
| Metrica -> SkillCorner (transfer, SkillCorner never used to fit) | 0.306 / 0.726 | 0.399 / 0.536 |

`temp` was selected on a sweep over these two games, so these numbers are optimistic. The gain is roughly
halved when transferring, and macroF1 drops.

TODO: result on the public five-minute window, reported only as a pipeline-correctness check and never used
to choose among variants.

## 4. Possession rules

- A possession is a maximal run of H or A; dead-ball stretches split possessions.
- External labels follow the competition definition: control events (pass, carry, shot, recovery, set piece) set
  the team; challenges and lost balls do not change it by themselves ("control, not touch"). Dead ball runs from
  a ball-out, foul or goal to the next restart. SkillCorner labels take the team of the covering phase of play
  and `D` where no phase exists.
- On the competition match the model has no rule for loose balls, challenges or restarts: they are handled only
  by the learned stickiness, which smooths them out. Dead-ball detection is limited to the pre-kickoff block.

## 5. Limitations and planned improvements

- **Not yet run on the competition data** (TODO: add final possession counts and any parser fixes).
- Event- and phase-derived external labels do not match the competition's tracking-based dead-ball definition.
- Only two external games; `temp` and the distance buckets are provisional.
- Weak dead-ball model; no set-piece or restart detection; Smart Tagging unused.
- Planned: dead-ball features (ball speed with a transferable calibration, proximity to lines, set-piece tags),
  more external games, joint fitting across games.

## 6. Compliance statement

- No hand annotation or hand correction of the provided match; no human labels about it.
- The private ground truth was never available to us. The public five-minute ground truth is used only to check
  that the scorer and pipeline run, not for training or tuning.
- Development used an AI coding assistant (Claude Code). It was configured to deny reads of the data folder and
  a hook blocks shell commands touching it; the match data was not given to it. It read the organiser's public
  code and documentation, our external datasets, and aggregate scoreboard values of the four published baselines.
- Match data is not committed or redistributed; `data/` is git-ignored.

## 7. Reproduction

```
pip install -r requirements.txt
python -m src.fit_params                      # optional: needs ext_open/metrica_game3 (public source above)
python -m src.run --data-dir /path/to/data    # writes outputs/submission.csv and validates it
python evaluation/validate_submission.py outputs/submission.csv
python -m unittest discover -s tests          # tests use synthetic data only
```

Python 3.9+ (unit tests run on 3.13), numpy and pandas only. The pipeline has no randomness, so it is expected
to regenerate the same CSV from the repository and the provided data. TODO: confirm by comparing SHA-256 of two
runs on the competition machine.
