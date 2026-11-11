"""End-to-end pipeline: competition ATD -> possession CSV. ONE command:

    python -m src.run [--data-dir PATH] [--out outputs/submission.csv]

Run it on the machine that holds the match data (DGX Spark). Data root: --data-dir, else the
MTAH_DATA_DIR environment variable, else ./data (organiser convention in games.py).
Prints aggregates only; never rows of data.

Steps: read ATD -> per-frame evidence -> Viterbi (HMM parameters from external games, models/hmm_params.json)
inside the graded halves -> D before the second-half kickoff -> X elsewhere -> frame,label CSV.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .external.common import REPO_ROOT, Tracking, load_possession_metrics
from .model.hmm import decode, extract_evidence
from .model.io import load_params

LETTERS = np.array(["H", "A", "D", "X"])
# Chosen on EXTERNAL games only (EXP-002 sweep, transfer Metrica -> SkillCorner); see PROVENANCE.md.
TEMP = 0.3
USE_SPEED = False


def predict(tr: Tracking, params, runs, forced_dead, temp: float = TEMP, use_speed: bool = USE_SPEED):
    """Pure function: Tracking -> int8 codes (H0 A1 D2 X3) of length n_frames."""
    codes = decode(extract_evidence(tr), params, runs, temp=temp, use_speed=use_speed, fill=3)
    for start, stop in forced_dead:
        codes[start:stop] = 2
    return codes


def write_csv(path, codes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))


def summarise(codes) -> str:
    counts = {str(l): int((codes == i).sum()) for i, l in enumerate(LETTERS)}
    alive = np.where(codes <= 1, codes, -1)
    changes = int(((alive[1:] != alive[:-1]) & (alive[1:] >= 0) & (alive[:-1] >= 0)).sum())
    return "label counts {} | H<->A switches {}".format(counts, changes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    ap.add_argument("--params", default=str(REPO_ROOT / "models" / "hmm_params.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "submission.csv"))
    args = ap.parse_args(argv)

    from .competition import load_competition

    params, meta = load_params(args.params)
    tr, runs, forced_dead = load_competition(args.data_dir)
    print("loaded tracking: {} frames, {} slots, ball visible on {:.1%} of frames".format(
        tr.n_frames, tr.players.shape[1], float((~np.isnan(tr.ball[:, 0])).mean())))
    codes = predict(tr, params, runs, forced_dead)
    write_csv(args.out, codes)
    print(summarise(codes))

    pm = load_possession_metrics()
    problem = pm.validate_submission(pm.read_label_csv(args.out), tr.n_frames)
    print("validation: {}".format("OK" if problem is None else "FAIL " + str(problem)))
    print("wrote {}".format(args.out))
    return 0 if problem is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
