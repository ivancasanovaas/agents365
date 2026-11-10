# AGENTS.md: More than a Hack 2027, Phase 1 (Ball Possession)

Portable instructions for any coding agent working in this repository.

## Goal
For every frame of one LaLiga match (BAR-ATH, 199,319 frames at 25 fps) output a label:
`H` home possession, `A` away possession, `D` dead ball, `X` abstain.
Deliverable CSV: `frame,label`, exactly 199,319 rows, frames 0..199318.
Ranking metric: `spellF1` (possession-level). Tie-break: `macroF1`. Baseline to beat: spellF1 0.2433 (smooth50).
Deadline: 2026-11-17.

## HARD RULES (never break)
1. **Never read, list, print or run anything that outputs content of `data/`** (tracking, video, Smart
   Tagging, homography, ground truth). The T&C forbid sending competition data to cloud LLMs.
   If a task needs it: STOP and give the human the command to run locally.
2. Scripts that touch data must print **aggregates only** (counts, metrics, paths), never rows.
3. **No hand annotation / hand correction** of this match, and no frame-specific rules.
4. The 5-minute public GT (`evaluation/gt/`) is for pipeline sanity checks only: never train,
   tune thresholds or pick between variants with it.
5. Every threshold/constant needs a documented origin in `PROVENANCE.md` (FIFA Laws, physics,
   other-game open data). Never "tuned by eye" or "tuned on the public GT".
6. Any external model, dataset or API must be accessible to the organisers and documented.
7. Never commit data or derived frame-level exports. The repo must regenerate the CSV with one command.

## Output-reading policy
- Allowed to read: `evaluation/scoreboard.csv` (aggregate metrics).
- NEVER read or report `evaluation/reports/*_transitions.csv` or `evaluation/reports/*.json` of OUR
  predictions: they expose frame-level transitions that could encode GT-derived information.
  The user reads those locally if needed.

## Architecture / execution environment
- Match data lives ONLY on the DGX Spark (Linux). It never leaves it and is never read by the agent.
- Code must be cross-platform: `pathlib` only, no hard-coded Windows paths, no shell-specific tricks.
- Data root comes from `MTAH_DATA_DIR` (or `data/`); never hard-code it.
- A single command must run the whole pipeline end to end (planned: `python -m src.run`), and it
  must print aggregates only.
- Development and tests on this machine use synthetic and external open data only.

## Layout
- `evaluation/`: official scorer (`run_evaluation.py`) and validator (`validate_submission.py`). Do not modify.
- `starter/`: organiser loaders and baseline. Reuse, do not edit in place.
- `src/`: our pipeline (to be created). `outputs/`: generated files (gitignored).
- `PROVENANCE.md`: origin of every threshold. `.claude/`: agent config, skills, hooks.

## Commands
```
pip install -r requirements.txt
python evaluation/run_evaluation.py --bootstrap 1000     # score everything in evaluation/results/
python evaluation/validate_submission.py <file>.csv      # run before every submission
```
Set `MTAH_DATA_DIR` to the data folder if it lives outside the repo.

## Conventions
- Fixed seeds, pinned dependencies, one command regenerates the final CSV.
- Differences below ~0.04 spellF1 are noise; compare variants with `--bootstrap`.
- Graded frames: 11,185-90,651 and 110,808-190,197. Frames 110,808-113,741 are graded and are `D`.
- `X` outside graded frames is free; inside a possession it is free for spellF1; at hand-overs, guess.
