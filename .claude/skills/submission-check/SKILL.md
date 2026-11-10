---
name: submission-check
description: Use before delivering the possession CSV. Validates format, graded-window semantics and reproducibility without reading competition data.
---

# submission-check

Commands are run by the USER locally (they touch only our CSV, not `data/`); the agent may run them
only on files in `evaluation/results/` or `outputs/`.

1. `python evaluation/validate_submission.py <file>.csv` must print OK (exactly 199,319 rows, frames 0..199318, labels H/A/D/X, no blanks).
2. File name is `<team_name>.csv`; UTF-8; header exactly `frame,label`; no index column; ~1.9 MB.
3. Semantics (aggregate checks only):
   - frames 110,808-113,741 (pre second-half kickoff) are `D`;
   - `X` only where intended (ungraded zones, or short in-possession gaps);
   - number of possessions is plausible (not hundreds of times the real count; compare with baseline_smooth50 count).
4. `python evaluation/run_evaluation.py --bootstrap 1000` runs without SKIPPED lines.
5. Reproducibility: re-running the documented command yields a byte-identical CSV (compare SHA-256).
6. Repo is public-ready: no `data/`, no frame-level exports, pinned dependencies, run instructions.
7. Technical document exists as `<team_name>.pdf` or `.md` (see tech-report skill).

Output a checklist with PASS/FAIL per item.
