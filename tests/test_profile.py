"""Tests for src/profile_datasets.py. Synthetic data only.

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.common import FPS, Tracking  # noqa: E402
from src.external.synthetic import make_match  # noqa: E402
from src.local_analyst import MAX_CHARS, MAX_LINE, check_loopback, read_aggregate  # noqa: E402
from src.profile_datasets import _run_bounds, _runs_in_graded, build_profile, render  # noqa: E402


def _setup(minutes=6):
    tr, _ = make_match(0, minutes=minutes)
    n = tr.n_frames
    ball = tr.ball.copy()
    ball[200:260] = np.nan            # a 60-frame gap
    ball[1000:1100] = [-3.0, 10.0]    # 100 frames outside the pitch
    ball[2000:2100] = [50.0, 30.0]    # 100 frames stationary
    tr = Tracking(ball=ball, players=tr.players, slot_team=tr.slot_team)
    runs = [(0, n // 2), (n // 2 + 500, n)]
    events = [
        {"code": "SET PIECES", "start_frame": 100, "end_frame": 220, "tags": {"Team": "N/A", "Type": "Throw-in"}},
        {"code": "BUILD UP", "start_frame": 300, "end_frame": 900, "tags": {"Team": "Barcelona"}},
    ]
    codes = np.zeros(n, dtype=np.int8)
    codes[n // 3:n // 2] = 1
    codes[n // 2:n // 2 + 500] = 3
    codes[2000:2100] = 2
    return tr, runs, events, codes


class RunHelperTests(unittest.TestCase):
    def test_run_bounds(self):
        starts, lengths = _run_bounds(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool))
        self.assertEqual(starts.tolist(), [1, 4, 7])
        self.assertEqual(lengths.tolist(), [2, 1, 3])

    def test_runs_never_cross_halves(self):
        mask = np.ones(100, dtype=bool)
        _, lengths = _runs_in_graded(mask, [(0, 40), (60, 100)])
        self.assertEqual(sorted(lengths.tolist()), [40, 40])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.runs, self.events, self.codes = _setup()
        self.text = render(build_profile(self.tr, self.runs, self.events, self.codes))

    def test_has_all_sections(self):
        for title in ("Structure", "Nulls and ranges", "Ball gaps (graded)", "Ball speed", "Ball out of bounds",
                      "Stillness", "Ball-player proximity", "Dead-ball proxy (descriptive)",
                      "Smart Tagging", "Our submission (aggregates)"):
            self.assertIn("## " + title, self.text)

    def test_is_an_aggregate_by_the_local_analyst_guard(self):
        self.assertLess(len(self.text), MAX_CHARS)
        self.assertLessEqual(max(len(l) for l in self.text.splitlines()), MAX_LINE)

    def test_known_injected_events_are_seen(self):
        self.assertIn("missing-ball runs inside graded halves", self.text)
        outside = [l for l in self.text.splitlines() if l.startswith("- outside runs")]
        self.assertTrue(outside and "count=" in outside[0] and "count=0" not in outside[0])
        still = [l for l in self.text.splitlines() if l.startswith("- ball stationary")]
        self.assertTrue(still and "count=0" not in still[0])

    def test_no_raw_values_leak(self):
        # no coordinate-like lists and no frame-index lists: every number sits next to a label
        for line in self.text.splitlines():
            self.assertNotIn("[[", line)
            self.assertNotIn("array(", line)

    def test_works_without_optional_parts(self):
        text = render(build_profile(self.tr, self.runs))
        self.assertNotIn("Smart Tagging", text)
        self.assertNotIn("Our submission", text)

    def test_deterministic(self):
        again = render(build_profile(self.tr, self.runs, self.events, self.codes))
        self.assertEqual(self.text, again)

    def test_visible_share_matches_direct_count(self):
        n = self.tr.n_frames
        graded = sum(hi - lo for lo, hi in self.runs)
        g = np.zeros(n, dtype=bool)
        for lo, hi in self.runs:
            g[lo:hi] = True
        share = 100.0 * (np.isfinite(self.tr.ball[:, 0]) & g).sum() / graded
        self.assertIn("graded {:.2f}%".format(share), self.text)


class LocalAnalystGuardTests(unittest.TestCase):
    def test_loopback_only(self):
        check_loopback("http://127.0.0.1:11435")
        check_loopback("http://localhost:11435")
        for bad in ("http://192.168.1.57:11434", "https://api.example.com", "http://0.0.0.0:11434"):
            with self.assertRaises(SystemExit):
                check_loopback(bad)

    def test_rejects_raw_looking_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.md"
            p.write_text("a" * (MAX_LINE + 1), encoding="utf-8")
            with self.assertRaises(SystemExit):
                read_aggregate(p)


if __name__ == "__main__":
    unittest.main()
