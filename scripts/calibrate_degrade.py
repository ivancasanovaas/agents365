"""Calibrate the degrader against AGGREGATE competition statistics (see PROVENANCE.md). External data only.

Run from the repo root:  python scripts/calibrate_degrade.py

Competition targets come from outputs/profile.md (aggregates printed by src/profile_datasets.py):
ball step p10/50/90/99 = 1.0/5.1/14.3/24.6 m/s, 0.23% of steps > 40 m/s,
median-player step speed p10/50/90/99 = 0.56/1.34/3.07/4.75 m/s.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external.common import FPS  # noqa: E402
from src.external.degrade import DegradeParams, calibrated_params, degrade  # noqa: E402
from src.external.metrica import load_game3  # noqa: E402
from src.model.hmm import EXTRA_NSYM, extract_extra  # noqa: E402

clean, gt, info = load_game3()


def stats(tr, tag):
    step = np.linalg.norm(np.diff(tr.ball, axis=0), axis=1) * FPS
    s = step[np.isfinite(step)]
    q = np.quantile(s, [0.1, 0.5, 0.9, 0.99])
    disp = np.linalg.norm(np.diff(tr.players, axis=0), axis=2) * FPS
    crowd = np.nanmedian(np.where(np.isfinite(disp), disp, np.nan), axis=1)
    cq = np.nanquantile(crowd, [0.1, 0.5, 0.9, 0.99])
    print("{:28s} ball step p10/50/90/99 = {:.1f}/{:.1f}/{:.1f}/{:.1f}  >40: {:.2f}%  "
          "crowd step p10/50/90/99 = {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
              tag, q[0], q[1], q[2], q[3], 100 * (s > 40).mean(), cq[0], cq[1], cq[2], cq[3]))


stats(clean, "clean")
for corr in (0.0, 0.9, 0.99, 0.999):
    stats(degrade(clean, 1, DegradeParams(noise_corr=corr, p_ball_glitch_per_min=1.7)),
          "corr={} glitch=1.7".format(corr))
stats(degrade(clean, 1, calibrated_params()), "calibrated (used)")
print("COMPETITION (profile): ball step p10/50/90/99 = 1.0/5.1/14.3/24.6  >40: 0.23%  "
      "crowd step p10/50/90/99 = 0.56/1.34/3.07/4.75")

# Symbol shares of the EXP-003 streams on the calibrated degraded game, to compare with the
# "EXP-003 stream distributions (graded)" section of outputs/profile.md (competition).
tr = degrade(clean, 1, calibrated_params())
for name, sym in extract_extra(tr, ("out", "crowd")).items():
    valid = sym >= 0
    print("stream {}: valid share {:.3f} | symbol shares {}".format(
        name, valid.mean(), " ".join("{:.3f}".format((sym[valid] == k).mean()) for k in range(EXTRA_NSYM[name]))))
