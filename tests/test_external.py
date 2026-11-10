"""Unit tests for the external bench. stdlib unittest only; synthetic data only.

Run:  python -m unittest discover -s tests -v
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.baseline import nearest_player_codes, smooth_codes  # noqa: E402
from src.external.common import A, D, H, X, Tracking, load_possession_metrics  # noqa: E402
from src.external.degrade import DegradeParams, degrade  # noqa: E402
from src.external.labels import labels_from_events  # noqa: E402
from src.external.synthetic import make_match  # noqa: E402


def _long_match(seed=0, minutes=20):
    return make_match(seed, minutes=minutes)


class DegradeTests(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        clean, _ = _long_match(minutes=3)
        a, b = degrade(clean, 1), degrade(clean, 1)
        c = degrade(clean, 2)
        np.testing.assert_array_equal(a.ball, b.ball)
        np.testing.assert_array_equal(a.players, b.players)
        self.assertFalse(np.array_equal(np.isnan(a.ball), np.isnan(c.ball)))

    def test_does_not_mutate_input(self):
        clean, _ = _long_match(minutes=2)
        ball0, players0 = clean.ball.copy(), clean.players.copy()
        degrade(clean, 3)
        np.testing.assert_array_equal(clean.ball, ball0)
        np.testing.assert_array_equal(clean.players, players0)

    def test_total_ball_missing_hits_target(self):
        clean, _ = _long_match(minutes=30)
        for target in (0.2, 0.41):
            rates = [np.isnan(degrade(clean, s, DegradeParams(p_ball_missing=target)).ball[:, 0]).mean()
                     for s in range(4)]
            self.assertAlmostEqual(float(np.mean(rates)), target, delta=0.04)

    def test_total_target_counts_existing_gaps(self):
        clean, _ = _long_match(minutes=30)
        ball = clean.ball.copy()
        ball[: len(ball) // 3] = np.nan  # input already misses ~33%
        pre = Tracking(ball=ball, players=clean.players, slot_team=clean.slot_team)
        rate = np.mean([np.isnan(degrade(pre, s).ball[:, 0]).mean() for s in range(4)])
        self.assertAlmostEqual(float(rate), 0.41, delta=0.05)

    def test_never_below_existing_missing(self):
        clean, _ = _long_match(minutes=5)
        ball = clean.ball.copy()
        ball[: int(len(ball) * 0.6)] = np.nan
        pre = Tracking(ball=ball, players=clean.players, slot_team=clean.slot_team)
        out = degrade(pre, 0)
        self.assertGreaterEqual(np.isnan(out.ball[:, 0]).mean(), 0.6 - 1e-9)

    def test_team_slots_stay_a_permutation(self):
        clean, _ = _long_match(minutes=3)
        out = degrade(clean, 5)
        self.assertEqual(sorted(out.slot_team.tolist()), sorted(clean.slot_team.tolist()))


class BaselineTests(unittest.TestCase):
    def test_nearest_player_team(self):
        players = np.array([[[0.0, 0.0], [50.0, 0.0]]])
        tr = Tracking(ball=np.array([[1.0, 0.0]]), players=players,
                      slot_team=np.array([0, 1], dtype=np.int8))
        self.assertEqual(nearest_player_codes(tr)[0], 0)
        tr.ball[0] = [np.nan, np.nan]
        self.assertEqual(nearest_player_codes(tr)[0], -1)

    def test_no_ball_modes(self):
        codes = np.array([0, 0, -1, -1, -1, -1, -1, -1, 1, 1], dtype=np.int8)
        d = smooth_codes(codes, 1, "D")
        x = smooth_codes(codes, 1, "X")
        self.assertEqual(d[4], D)
        self.assertEqual(x[4], X)
        np.testing.assert_array_equal(d[[0, 1, 8, 9]], x[[0, 1, 8, 9]])
        with self.assertRaises(ValueError):
            smooth_codes(codes, 1, "Z")

    def test_matches_organiser_smooth(self):
        path = ROOT / "starter" / "baseline.py"
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "starter"))
        try:
            spec = importlib.util.spec_from_file_location("org_baseline", path)
            org = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(org)
        except Exception as exc:  # organiser module may need optional deps
            self.skipTest("organiser baseline not importable: {}".format(exc))
        rng = np.random.default_rng(0)
        codes = rng.choice([-1, 0, 1], size=2000, p=[0.5, 0.25, 0.25]).astype(np.int8)
        for window in (0, 25, 50, 125):
            expected = org.smooth(codes, window)
            ours = smooth_codes(codes, window, "D")
            mapped = np.array(["H", "A", "D"])[ours]
            np.testing.assert_array_equal(mapped, expected)


class LabelAndMetricTests(unittest.TestCase):
    def test_labels_from_events(self):
        codes = labels_from_events(
            n_frames=100,
            control_events=[(10, 0), (40, 1)],
            dead_intervals=[(60, 70)],
            graded_runs=[(5, 90)],
        )
        self.assertTrue((codes[:5] == X).all())
        self.assertTrue((codes[5:10] == D).all())      # before the first event
        self.assertTrue((codes[10:40] == H).all())
        self.assertTrue((codes[40:60] == A).all())
        self.assertTrue((codes[60:70] == D).all())
        self.assertTrue((codes[70:90] == A).all())
        self.assertTrue((codes[90:] == X).all())

    def test_perfect_prediction_scores_one(self):
        pm = load_possession_metrics()
        _, gt = make_match(1, minutes=5)
        res = pm.evaluate(gt, gt.copy(), bootstrap_n=0)
        self.assertAlmostEqual(res["spell_f1_mean"], 1.0)
        self.assertAlmostEqual(res["macro_f1"], 1.0)


class HmmTests(unittest.TestCase):
    def test_viterbi_smooths_isolated_noise(self):
        from src.model.hmm import viterbi
        truth = np.array([0] * 50 + [1] * 50)
        ll = np.zeros((100, 3))
        ll[np.arange(100), truth] = 1.0
        ll[20, :] = [0.0, 1.0, 0.0]  # one contradicting frame inside the home spell
        trans = np.log(np.array([[0.98, 0.01, 0.01]] * 3) * 0 + np.where(np.eye(3) > 0, 0.98, 0.01))
        path = viterbi(ll, trans, np.log(np.ones(3) / 3))
        np.testing.assert_array_equal(path, truth)

    def test_decode_fills_outside_runs_and_fit_shapes(self):
        from src.model.hmm import decode, extract_evidence, fit
        clean, gt = make_match(2, minutes=2)
        ev = extract_evidence(clean)
        mask = gt != X
        params = fit(ev, gt, mask)
        self.assertEqual(params.log_trans.shape, (3, 3))
        np.testing.assert_allclose(np.exp(params.log_trans).sum(axis=1), 1.0)
        out = decode(ev, params, [(200, 1000)])
        self.assertTrue((out[:200] == X).all() and (out[1000:] == X).all())
        self.assertTrue(((out[200:1000] >= 0) & (out[200:1000] <= 2)).all())


class PipelineTests(unittest.TestCase):
    def test_read_atd_format_and_alignment(self):
        import tempfile
        from src.competition import read_atd
        text = (
            "1:0.5,0.5,1.0;NaN;0.25,0.5,0.0:0.5,0.25,2.0\n"
            "2:0.1,0.2,0.0;0.3,0.4,0.0;NaN:NaN\n"
            "3:garbage\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x_atd.txt"
            p.write_text(text)
            ball, players = read_atd(p, n_frames=4, pitch=(100.0, 50.0), n_slots=3)
        np.testing.assert_allclose(ball[0], [50.0, 12.5])        # frame_id 1 -> row 0, metres
        self.assertTrue(np.isnan(ball[1]).all() and np.isnan(ball[3]).all())
        np.testing.assert_allclose(players[0, 2], [25.0, 25.0])
        self.assertTrue(np.isnan(players[0, 1]).all())            # missing slot -> NaN
        np.testing.assert_allclose(players[1, 1], [30.0, 20.0])
        self.assertTrue(np.isnan(players[2:]).all())              # malformed / absent lines stay NaN

    def test_params_roundtrip_and_predict(self):
        import tempfile
        from src.model.hmm import extract_evidence, fit
        from src.model.io import load_params, save_params
        from src.run import predict, write_csv
        clean, gt = make_match(3, minutes=2)
        params = fit(extract_evidence(clean), gt, gt != X)
        with tempfile.TemporaryDirectory() as tmp:
            save_params(Path(tmp) / "p.json", params, {"note": "test"})
            loaded, meta = load_params(Path(tmp) / "p.json")
            np.testing.assert_allclose(loaded.log_trans, params.log_trans)
            n = clean.n_frames
            runs = [(100, 1500), (1800, n)]
            codes = predict(clean, loaded, runs, forced_dead=[(1800, 1900)])
            self.assertTrue((codes[:100] == X).all())
            self.assertTrue((codes[1500:1800] == X).all())
            self.assertTrue((codes[1800:1900] == D).all())
            out = Path(tmp) / "s.csv"
            write_csv(out, codes)
            lines = out.read_text().splitlines()
            self.assertEqual(lines[0], "frame,label")
            self.assertEqual(len(lines), n + 1)
            self.assertEqual(lines[1], "0,X")


if __name__ == "__main__":
    unittest.main()
