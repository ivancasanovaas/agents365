"""Tests for the 4-state HMM with a dead state that remembers the last possessor (EXP-015). Synthetic data only."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.hmm4 import decode4, fit_transitions4, to_four_state  # noqa: E402


class FourStateTests(unittest.TestCase):
    def test_conversion_forward_fills_the_last_possessor(self):
        H, A, D, X = 0, 1, 2, 3
        codes = np.array([X, D, D, H, H, D, D, A, D, X], dtype=np.int8)
        out = to_four_state(codes, [(1, 9)])
        # frames 1-2: D before any possession -> undefined; 3-4 H; 5-6 D after H (=2); 7 A; 8 D after A (=3)
        self.assertEqual(out.tolist(), [-1, -1, -1, 0, 0, 2, 2, 1, 3, -1])

    def test_learns_that_a_restart_goes_to_the_opponent(self):
        # synthetic labels: possessions alternate H, D, A, D, H, ... (the restart always goes to the other team)
        block = [0] * 30 + [2] * 10 + [1] * 30 + [2] * 10
        codes = np.array(block * 20, dtype=np.int8)
        lt, lp = fit_transitions4([codes], [[(0, len(codes))]])
        p = np.exp(lt)
        self.assertGreater(p[2, 1], 10 * p[2, 0])        # D_H -> A far more likely than D_H -> H
        self.assertGreater(p[3, 0], 10 * p[3, 1])        # D_A -> H far more likely than D_A -> A
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertAlmostEqual(float(np.exp(lp).sum()), 1.0)

    def test_decode_uses_the_memory_when_the_evidence_is_neutral(self):
        block = [0] * 30 + [2] * 10 + [1] * 30 + [2] * 10
        codes = np.array(block * 20, dtype=np.int8)
        lt, lp = fit_transitions4([codes], [[(0, len(codes))]])
        # evidence: clear H for 40 frames, clear D for 10, then completely neutral for 40 frames
        n = 90
        ll3 = np.zeros((n, 3))
        ll3[:40, 0], ll3[:40, 1], ll3[:40, 2] = 3.0, -3.0, -3.0
        ll3[40:50, 2], ll3[40:50, 0], ll3[40:50, 1] = 3.0, -3.0, -3.0
        out = decode4(ll3, lt, lp, [(0, n)])
        self.assertTrue((out[:40] == 0).all())
        self.assertTrue((out[40:50] == 2).all())
        self.assertEqual(int(out[60]), 1)                # neutral evidence after a home possession + stop: the restart is away

    def test_output_labels_and_fill(self):
        ll3 = np.log(np.full((50, 3), 1 / 3))
        lt, lp = fit_transitions4([np.array([0] * 20 + [2] * 5 + [1] * 20)], [[(0, 45)]])
        out = decode4(ll3, lt, lp, [(5, 40)])
        self.assertTrue(set(np.unique(out[5:40])) <= {0, 1, 2})
        self.assertTrue((out[:5] == 3).all() and (out[40:] == 3).all())


if __name__ == "__main__":
    unittest.main()
