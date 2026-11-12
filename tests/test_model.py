"""Tests for the dead-ball features, the extra HMM streams and the degrader options. Synthetic data only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.common import FPS, PITCH_M, Tracking  # noqa: E402
from src.external.degrade import DegradeParams, degrade  # noqa: E402
from src.external.synthetic import make_match  # noqa: E402
from src.model import features as F  # noqa: E402
from src.model.hmm import ALL_EXTRA, EXTRA_NSYM, decode, extract_evidence, fit  # noqa: E402
from src.model.io import load_params, save_params  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_outside_mask_and_run_lengths(self):
        ball = np.array([[10, 10], [-1, 5], [-2, 5], [20, 20], [np.nan, np.nan], [200, 5]], dtype=float)
        mask = F.outside_mask(ball)
        self.assertEqual(mask.tolist(), [False, True, True, False, False, True])
        self.assertEqual(F.run_length_per_frame(mask).tolist(), [0, 2, 2, 0, 0, 1])

    def test_crowd_speed_still_vs_moving(self):
        n, p = 200, 6
        still = np.full((n, p, 2), 30.0)
        moving = np.zeros((n, p, 2))
        moving[:, :, 0] = np.arange(n)[:, None] * (2.0 / FPS)       # 2 m/s along x
        self.assertLess(np.nanmax(F.crowd_speed(still)), 1e-9)
        np.testing.assert_allclose(np.nanmedian(F.crowd_speed(moving)), 2.0, atol=1e-6)

    def test_crowd_speed_ignores_a_single_swapping_slot(self):
        n, p = 200, 11
        pos = np.full((n, p, 2), 30.0)
        pos[:, 0, 0] += np.where(np.arange(n) % 2 == 0, 40.0, 0.0)   # one slot jumping around
        self.assertLess(np.nanmax(F.crowd_speed(pos)), 1e-9)          # median is unaffected

    def test_gap_edge_distance(self):
        length, width = PITCH_M
        ball = np.array([[length - 1.0, 30.0], [np.nan, np.nan], [np.nan, np.nan], [50.0, 30.0]])
        d = F.gap_edge_distance(ball)
        self.assertTrue(np.isnan(d[0]) and np.isnan(d[3]))
        self.assertAlmostEqual(d[1], 1.0, places=6)
        self.assertAlmostEqual(d[2], 1.0, places=6)


def _case(minutes=8, seed=0):
    tr, gt = make_match(seed, minutes=minutes)
    return degrade(tr, seed=1, params=DegradeParams()), gt


class HmmStreamTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.gt = _case()
        self.n = self.tr.n_frames
        self.runs = [(0, self.n)]

    def test_evidence_symbols_in_range(self):
        ev = extract_evidence(self.tr, streams=ALL_EXTRA)
        for name, sym in ev.extra.items():
            self.assertEqual(len(sym), self.n)
            self.assertGreaterEqual(sym.min(), -1)
            self.assertLess(sym.max(), EXTRA_NSYM[name])

    def test_unknown_stream_rejected(self):
        with self.assertRaises(ValueError):
            extract_evidence(self.tr, streams=("nope",))

    def test_fit_learns_requested_streams_only(self):
        ev = extract_evidence(self.tr, streams=("out", "crowd"))
        params = fit(ev, self.gt, self.gt != 3)
        self.assertEqual(sorted(params.log_extra), ["crowd", "out"])
        for name, table in params.log_extra.items():
            np.testing.assert_allclose(np.exp(table).sum(axis=1), 1.0)

    def test_streams_empty_equals_legacy_decode(self):
        ev0 = extract_evidence(self.tr)                       # legacy: no extras
        ev1 = extract_evidence(self.tr, streams=ALL_EXTRA)
        p0 = fit(ev0, self.gt, self.gt != 3)
        p1 = fit(ev1, self.gt, self.gt != 3)
        legacy = decode(ev0, p0, self.runs, temp=0.3, use_speed=False)
        gated = decode(ev1, p1, self.runs, temp=0.3, use_speed=False, streams=())
        np.testing.assert_array_equal(legacy, gated)

    def test_decode_with_streams_runs_and_is_deterministic(self):
        ev = extract_evidence(self.tr, streams=("out", "crowd"))
        params = fit(ev, self.gt, self.gt != 3)
        a = decode(ev, params, self.runs, temp=0.3, use_speed=False, streams=("out", "crowd"))
        b = decode(ev, params, self.runs, temp=0.3, use_speed=False, streams=("out", "crowd"))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(set(np.unique(a)) <= {0, 1, 2})

    def test_params_roundtrip_and_legacy_file_loads(self):
        ev = extract_evidence(self.tr, streams=("out",))
        params = fit(ev, self.gt, self.gt != 3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            save_params(path, params, {"x": 1})
            loaded, meta = load_params(path)
            np.testing.assert_allclose(loaded.log_extra["out"], params.log_extra["out"])
            payload = json.loads(path.read_text())
            payload.pop("log_extra")                           # a pre-EXP-003 file
            path.write_text(json.dumps(payload))
            legacy, _ = load_params(path)
            self.assertEqual(legacy.log_extra, {})


class DegradeOptionTests(unittest.TestCase):
    def test_defaults_unchanged_by_new_options(self):
        clean, _ = make_match(0, minutes=3)
        a = degrade(clean, 5, DegradeParams())
        b = degrade(clean, 5, DegradeParams(noise_corr=0.0, p_ball_glitch_per_min=0.0))
        np.testing.assert_array_equal(a.ball, b.ball)
        np.testing.assert_array_equal(a.players, b.players)

    def test_glitches_create_fast_outliers(self):
        clean, _ = make_match(0, minutes=10)
        def fast(tr):
            step = np.linalg.norm(np.diff(tr.ball, axis=0), axis=1) * FPS
            return int(np.nansum(step > 40.0))
        base = degrade(clean, 7, DegradeParams(noise_corr=0.999))
        glitchy = degrade(clean, 7, DegradeParams(noise_corr=0.999, p_ball_glitch_per_min=2.0))
        self.assertGreater(fast(glitchy), fast(base) + 5)

    def test_correlated_noise_is_smoother(self):
        clean, _ = make_match(0, minutes=3)
        iid = degrade(clean, 3, DegradeParams(noise_corr=0.0))
        ar = degrade(clean, 3, DegradeParams(noise_corr=0.99))
        def jitter(tr):
            return float(np.nanmedian(np.linalg.norm(np.diff(tr.players, axis=0), axis=2)))
        self.assertLess(jitter(ar), jitter(iid))


if __name__ == "__main__":
    unittest.main()
