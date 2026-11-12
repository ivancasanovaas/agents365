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


class DurationTests(unittest.TestCase):
    def test_scale_ha_only_touches_ha_transitions(self):
        from src.model.hmm import scale_ha
        tr, gt = _case(minutes=6)
        p = fit(extract_evidence(tr), gt, gt != 3)
        q = scale_ha(p, 0.25)
        a, b = np.exp(p.log_trans), np.exp(q.log_trans)
        np.testing.assert_allclose(b.sum(axis=1), 1.0)
        self.assertLess(b[0, 1], a[0, 1])
        self.assertLess(b[1, 0], a[1, 0])
        # ratios among the untouched entries of a row are preserved (H row: stay vs go to D)
        np.testing.assert_allclose(b[0, 0] / b[0, 2], a[0, 0] / a[0, 2])
        np.testing.assert_allclose(scale_ha(p, 1.0).log_trans, p.log_trans)

    def test_min_duration_filter_cases(self):
        from src.model.postfilter import min_duration_filter, spell_lengths
        H, A, D = 0, 1, 2
        seq = np.array([H] * 10 + [A] * 2 + [H] * 10 + [D] * 5 + [A] * 2 + [D] * 5 + [A] * 8 + [A] * 0, dtype=np.int8)
        out = min_duration_filter(seq, 5, [(0, len(seq))])
        self.assertEqual(out[10:12].tolist(), [H, H])                 # short A between H runs absorbed by previous
        self.assertEqual(out[27:29].tolist(), [D, D])                 # short A between two D becomes D
        self.assertEqual(out[34:].tolist(), [A] * 8)                  # long runs untouched
        self.assertTrue(spell_lengths(out, [(0, len(out))]).min() >= 5)

    def test_filter_leaves_outside_runs_and_does_not_mutate(self):
        from src.model.postfilter import min_duration_filter
        seq = np.array([3, 3, 0, 1, 0, 0, 0, 0, 3, 3], dtype=np.int8)
        before = seq.copy()
        out = min_duration_filter(seq, 3, [(2, 8)])
        np.testing.assert_array_equal(seq, before)
        self.assertEqual(out[:2].tolist(), [3, 3])
        self.assertEqual(out[8:].tolist(), [3, 3])
        self.assertEqual(out[2:8].tolist(), [0] * 6)

    def test_duration_quantile_from_labels(self):
        from src.model.postfilter import spell_duration_quantile
        gt = np.array([0] * 10 + [2] * 3 + [1] * 30 + [0] * 20, dtype=np.int8)
        q = spell_duration_quantile(gt, [(0, len(gt))], 0.5)
        self.assertEqual(q, 20)                                       # lengths 10, 30, 20 -> median 20


class PoolTests(unittest.TestCase):
    def test_concat_and_pooled_fit_sums_counts(self):
        from src.model.hmm import concat_evidence
        t1, g1 = _case(minutes=4, seed=0)
        t2, g2 = _case(minutes=5, seed=1)
        e1 = extract_evidence(t1, streams=("out", "crowd"))
        e2 = extract_evidence(t2, streams=("out", "crowd"))
        pooled = concat_evidence([e1, e2])
        self.assertEqual(len(pooled.team_sym), t1.n_frames + t2.n_frames)
        gt = np.concatenate([g1, g2])
        p = fit(pooled, gt, gt != 3)
        for table in p.log_extra.values():
            np.testing.assert_allclose(np.exp(table).sum(axis=1), 1.0)
        with self.assertRaises(ValueError):
            concat_evidence([e1, extract_evidence(t2, streams=("out",))])


class QuantileEdgeTests(unittest.TestCase):
    def test_edges_give_equal_share_buckets(self):
        tr, _ = _case(minutes=10)
        edges = F.crowd_edges_for_game(tr.players, [(0, tr.n_frames)])
        self.assertEqual(len(edges), 4)
        self.assertTrue(np.all(np.diff(edges) > 0))
        ev = extract_evidence(tr, streams=("crowd",), crowd_edges=edges)
        sym = ev.extra["crowd"]
        shares = [(sym[sym >= 0] == k).mean() for k in range(5)]
        np.testing.assert_allclose(shares, 0.2, atol=0.01)

    def test_edges_do_not_use_ungraded_frames(self):
        tr, _ = _case(minutes=6)
        n = tr.n_frames
        slow = tr.players.copy()
        slow[: n // 2] = slow[: n // 2, :, :][:1]               # first half frozen: speed 0 everywhere
        frozen = Tracking(ball=tr.ball, players=slow, slot_team=tr.slot_team)
        graded_only = F.crowd_edges_for_game(frozen.players, [(n // 2 + 100, n - 100)])
        whole = F.crowd_edges_for_game(frozen.players, [(0, n)])
        self.assertGreater(graded_only[0], whole[0])             # frozen half would drag the edges down if included

    def test_default_edges_unchanged(self):
        tr, _ = _case(minutes=4)
        a = extract_evidence(tr, streams=("crowd",))
        b = extract_evidence(tr, streams=("crowd",), crowd_edges=None)
        np.testing.assert_array_equal(a.extra["crowd"], b.extra["crowd"])


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

    def test_calibrated_gaps_match_profile_shape(self):
        from src.external.degrade import calibrated_params
        clean, _ = make_match(0, minutes=60)
        tr = degrade(clean, 11, calibrated_params())
        miss = ~np.isfinite(tr.ball[:, 0])
        padded = np.concatenate(([0], miss.astype(np.int8), [0]))
        d = np.diff(padded)
        lens = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / FPS
        self.assertAlmostEqual(float(miss.mean()), 0.41, delta=0.05)
        self.assertGreater(float(np.median(lens)), 6.0)        # bench v1 had a ~1.4 s median
        self.assertLess(float(np.median(lens)), 16.0)

    def test_correlated_noise_is_smoother(self):
        clean, _ = make_match(0, minutes=3)
        iid = degrade(clean, 3, DegradeParams(noise_corr=0.0))
        ar = degrade(clean, 3, DegradeParams(noise_corr=0.99))
        def jitter(tr):
            return float(np.nanmedian(np.linalg.norm(np.diff(tr.players, axis=0), axis=2)))
        self.assertLess(jitter(ar), jitter(iid))


if __name__ == "__main__":
    unittest.main()
