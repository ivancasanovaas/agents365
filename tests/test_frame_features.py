"""Tests for src/model/frame_features.py and src/model/cluster.py. Synthetic data only."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.common import FPS, PITCH_M, Tracking  # noqa: E402
from src.external.synthetic import make_match  # noqa: E402
from src.model.cluster import adjusted_rand, assign, kmeans, pca, standardise  # noqa: E402
from src.model.frame_features import (COLUMN_DOC, FULL_DOC, add_normalised, aggregate_table,  # noqa: E402
                                      attack_direction, build_frame_table)


def _controlled(n=300, k=4):
    """Two teams of k slots with exactly known geometry: home left block, away right block, ball at x=50."""
    pos = np.zeros((n, 2 * k, 2))
    team = np.repeat(np.array([0, 1], dtype=np.int8), k)
    ys = np.linspace(10, 58, k)
    pos[:, :k, 0], pos[:, :k, 1] = 20.0, ys                  # home x=20
    pos[:, k:, 0], pos[:, k:, 1] = 80.0, ys                  # away x=80
    ball = np.tile(np.array([50.0, 34.0]), (n, 1))
    return Tracking(ball=ball, players=pos, slot_team=team)


class FrameTableTests(unittest.TestCase):
    def test_every_column_is_documented_and_vice_versa(self):
        tr, _ = make_match(0, minutes=2)
        df = build_frame_table(tr, [(0, tr.n_frames)])
        self.assertEqual(list(df.columns), list(COLUMN_DOC))
        self.assertEqual(len(df), tr.n_frames)
        self.assertTrue(all(len(v) > 20 for v in FULL_DOC.values()))

    def test_known_geometry(self):
        tr = _controlled()
        df = build_frame_table(tr, [(0, tr.n_frames)])
        mid = df.iloc[100]
        self.assertEqual(int(mid["n_h"]), 4)
        self.assertAlmostEqual(mid["cx_h"], 20.0)
        self.assertAlmostEqual(mid["cx_a"], 80.0)
        self.assertAlmostEqual(mid["cy_h"], np.mean(np.linspace(10, 58, 4)))
        self.assertAlmostEqual(mid["sx_h"], 0.0)
        self.assertAlmostEqual(mid["dball_min_h"], np.hypot(30.0, np.min(np.abs(np.linspace(10, 58, 4) - 34.0))))
        self.assertAlmostEqual(mid["dball_cen_h"], np.hypot(30.0, 0.0), places=6)
        self.assertEqual(int(mid["near5_h"]), 0)
        # all four home slots are on the same x, ys 10,26,42,58: mean pairwise distance
        ys = np.linspace(10, 58, 4)
        pair = np.mean([abs(a - b) for i, a in enumerate(ys) for j, b in enumerate(ys) if i != j])
        self.assertAlmostEqual(mid["pair_h"], pair, places=6)
        self.assertAlmostEqual(mid["spd_mean_h"], 0.0)
        self.assertAlmostEqual(mid["crowd_speed"], 0.0)

    def test_moving_team_speed_and_centroid_velocity(self):
        tr = _controlled()
        tr.players[:, :4, 0] += np.arange(tr.n_frames)[:, None] * (3.0 / FPS)   # home advances at 3 m/s
        df = build_frame_table(tr)
        mid = df.iloc[150]
        self.assertAlmostEqual(mid["spd_mean_h"], 3.0, places=6)
        self.assertAlmostEqual(mid["cvx_h"], 3.0, places=6)
        self.assertAlmostEqual(mid["spd_mean_a"], 0.0, places=6)

    def test_hidden_ball_gives_nan_ball_relative_columns_only(self):
        tr = _controlled()
        tr.ball[:] = np.nan
        df = build_frame_table(tr)
        row = df.iloc[100]
        self.assertTrue(np.isnan(row["dball_min_h"]) and np.isnan(row["near5_h"]))
        self.assertEqual(int(row["ball_visible"]), 0)
        self.assertAlmostEqual(row["cx_h"], 20.0)

    def test_too_few_players_gives_nan_team_stats(self):
        tr = _controlled()
        tr.players[:, 1:4] = np.nan                           # only 1 home slot left
        row = build_frame_table(tr).iloc[100]
        self.assertEqual(int(row["n_h"]), 1)
        self.assertTrue(np.isnan(row["cx_h"]) and np.isnan(row["pair_h"]))
        self.assertAlmostEqual(row["cx_a"], 80.0)

    def test_half_and_ball_out(self):
        tr = _controlled()
        tr.ball[10:20] = [-2.0, 30.0]
        df = build_frame_table(tr, [(5, 100), (150, 250)])
        self.assertEqual(df["half"].iloc[0], 0)
        self.assertEqual(df["half"].iloc[50], 1)
        self.assertEqual(df["half"].iloc[200], 2)
        self.assertEqual(int(df["ball_out"].iloc[10:20].sum()), 10)

    def test_orientation_normalisation_flips_second_half(self):
        n = 200
        a = _controlled(n)
        b = _controlled(n)
        b.players[:, :, 0] = PITCH_M[0] - b.players[:, :, 0]  # second half: sides swapped (home on the right)
        b.ball[:, 0] = PITCH_M[0] - b.ball[:, 0]
        players = np.concatenate([a.players, b.players])
        ball = np.concatenate([a.ball, b.ball])
        tr = Tracking(ball=ball, players=players, slot_team=a.slot_team)
        df = build_frame_table(tr, [(0, n), (n, 2 * n)])
        self.assertEqual(attack_direction(df), {1: 1, 2: -1})
        dn = add_normalised(df)
        np.testing.assert_allclose(dn["cx_h_n"].iloc[[100, 300]], 20.0)
        np.testing.assert_allclose(dn["cx_a_n"].iloc[[100, 300]], 80.0)

    def test_aggregate_table_has_no_rows(self):
        tr, _ = make_match(0, minutes=2)
        df = build_frame_table(tr, [(0, tr.n_frames)])
        agg = aggregate_table(df, ["cx_h", "spd_mean_h"])
        self.assertEqual(list(agg.index), ["cx_h", "spd_mean_h"])
        self.assertLessEqual(len(agg), 2)
        self.assertIn("p50", agg.columns)

    def test_deterministic(self):
        tr, _ = make_match(1, minutes=2)
        a = build_frame_table(tr, [(0, tr.n_frames)])
        b = build_frame_table(tr, [(0, tr.n_frames)])
        self.assertTrue(a.equals(b))


class ClusterTests(unittest.TestCase):
    def _blobs(self):
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(0, 0.3, (200, 3)), rng.normal(6, 0.3, (200, 3))])
        y = np.repeat([0, 1], 200)
        return X, y

    def test_kmeans_separates_blobs_and_is_deterministic(self):
        X, y = self._blobs()
        l1, c1, i1 = kmeans(X, 2, seed=3)
        l2, c2, i2 = kmeans(X, 2, seed=3)
        np.testing.assert_array_equal(l1, l2)
        self.assertAlmostEqual(adjusted_rand(l1, y), 1.0)

    def test_adjusted_rand_bounds(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 3, 500)
        self.assertAlmostEqual(adjusted_rand(a, a), 1.0)
        self.assertAlmostEqual(adjusted_rand(a, (a + 1) % 3), 1.0)          # relabelling does not matter
        self.assertLess(abs(adjusted_rand(a, rng.integers(0, 3, 500))), 0.05)

    def test_pca_explained_variance(self):
        X, _ = self._blobs()
        Z, _, _ = standardise(X)
        comps, ratio = pca(Z, 2)
        self.assertEqual(comps.shape, (2, 3))
        self.assertGreater(ratio[0], 0.9)
        self.assertLessEqual(ratio.sum(), 1.0 + 1e-9)

    def test_assign_matches_nearest_centre(self):
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        X = np.array([[1.0, 1.0], [9.0, 9.0], [4.0, 4.0], [6.0, 6.0]])
        self.assertEqual(assign(X, centers).tolist(), [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
