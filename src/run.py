"""End-to-end pipeline: competition ATD -> possession CSV. ONE command:

    python -m src.run [--data-dir PATH] [--out outputs/submission.csv] [--model softmax|nb]

Run it on the machine that holds the match data (DGX Spark). Data root: --data-dir, else the
MTAH_DATA_DIR environment variable, else ./data (organiser convention in games.py).
Prints aggregates only; never rows of data.

Steps: read ATD -> per-frame collective features (tracking only) -> per-frame emissions from a softmax model trained on
EXTERNAL games (models/softmax_params.json, EXP-010) -> Viterbi with the transition matrix of models/hmm_params.json inside
the graded halves -> D before the second-half kickoff -> X elsewhere -> frame,label CSV.
`--model nb` runs the previous naive-Bayes model (variant C of EXP-003) for comparison.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .external.common import REPO_ROOT, Tracking, load_possession_metrics
from .model.frame_clf import build_features, decode_softmax
from .model.hmm import decode, extract_evidence
from .model.io import load_params, load_softmax

LETTERS = np.array(["H", "A", "D", "X"])
# Same temperature as the naive-Bayes variant; NOT tuned for the softmax (pre-registered in EXP-010).
TEMP = 0.3
USE_SPEED = False
# EXP-003 variant C streams, used only by --model nb.
STREAMS = ("out", "crowd")


def predict(tr: Tracking, params, runs, forced_dead, temp: float = TEMP, use_speed: bool = USE_SPEED,
            streams=STREAMS, softmax=None):
    """Pure function: Tracking -> int8 codes (H0 A1 D2 X3) of length n_frames.
    With `softmax` (a fitted SoftmaxModel) the emissions come from it; otherwise from the naive-Bayes streams."""
    if softmax is not None:
        X, _ = build_features(tr, runs)
        codes = decode_softmax(softmax, X, params.log_trans, params.log_prior, runs, temp)
    else:
        codes = decode(extract_evidence(tr, streams=streams), params, runs, temp=temp, use_speed=use_speed,
                       fill=3, streams=streams)
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
    ap.add_argument("--softmax", default=str(REPO_ROOT / "models" / "softmax_params.json"))
    ap.add_argument("--model", choices=["softmax", "nb"], default="softmax")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "submission.csv"))
    args = ap.parse_args(argv)

    from .competition import load_competition

    params, meta = load_params(args.params)
    softmax = None
    if args.model == "softmax":
        softmax, smeta = load_softmax(args.softmax)
        print("model: softmax emissions fitted on {} frames of external games".format(
            sum(smeta.get("n_frames", {}).values())))
    else:
        print("model: naive-Bayes streams {}".format(STREAMS))
    tr, runs, forced_dead = load_competition(args.data_dir)
    print("loaded tracking: {} frames, {} slots, ball visible on {:.1%} of frames".format(
        tr.n_frames, tr.players.shape[1], float((~np.isnan(tr.ball[:, 0])).mean())))
    codes = predict(tr, params, runs, forced_dead, softmax=softmax)
    write_csv(args.out, codes)
    print(summarise(codes))

    pm = load_possession_metrics()
    problem = pm.validate_submission(pm.read_label_csv(args.out), tr.n_frames)
    print("validation: {}".format("OK" if problem is None else "FAIL " + str(problem)))
    print("wrote {}".format(args.out))
    return 0 if problem is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
