"""Adapter test with a tiny synthetic file in the Metrica open-CSV layout. No real data."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.metrica_csv import load_game  # noqa: E402
from src.external.registry import available, load  # noqa: E402


def _write(dirpath: Path, n=40):
    stem = "Sample_Game_9"
    def tracking(team, with_ball):
        path = dirpath / "{}_RawTrackingData_{}_Team.csv".format(stem, team)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "", "", team, "", team, ""] + ([""] if with_ball else []))
            w.writerow(["", "", "", "1", "", "2", ""] + ([""] if with_ball else []))
            w.writerow(["Period", "Frame", "Time [s]", "Player1", "", "Player2", ""] + (["Ball", ""] if with_ball else []))
            for i in range(n):
                period = 1 if i < n // 2 else 2
                row = [period, i + 1, round((i + 1) * 0.04, 2), 0.2 + 0.001 * i, 0.5, 0.3, "NaN"]
                if with_ball:
                    row += [0.5, 0.5]
                w.writerow(row)
    tracking("Home", True)
    tracking("Away", True)          # the real Away file also ends with the ball columns
    with open(dirpath / "{}_RawEventsData.csv".format(stem), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame", "End Time [s]",
                    "From", "To", "Start X", "Start Y", "End X", "End Y"])
        w.writerow(["Home", "SET PIECE", "KICK OFF", 1, 1, 0.04, 0, 0, "Player1", "", "NaN", "NaN", "NaN", "NaN"])
        w.writerow(["Away", "PASS", "", 1, 10, 0.4, 12, 0.48, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5])
        w.writerow(["Away", "BALL OUT", "", 1, 15, 0.6, 15, 0.6, "Player1", "", 0.5, 0.5, 0.5, 0.5])
        w.writerow(["Home", "SET PIECE", "THROW IN", 1, 18, 0.72, 19, 0.76, "Player1", "", 0.1, 0.1, 0.1, 0.1])
        w.writerow(["Home", "PASS", "", 2, 25, 1.0, 27, 1.08, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5])


class MetricaCsvTests(unittest.TestCase):
    def test_parse_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d))
            tr, gt, info = load_game(9, Path(d))
        self.assertEqual(tr.n_frames, 40)
        self.assertEqual(tr.players.shape, (40, 4, 2))               # 2 home + 2 away slots
        self.assertEqual(tr.slot_team.tolist(), [0, 0, 1, 1])
        np.testing.assert_allclose(tr.ball[0], [52.5, 34.0])          # 0.5 * (105, 68)
        self.assertEqual(info["halves"], [(0, 20), (20, 40)])
        self.assertTrue(np.isnan(tr.players[0, 1, 1]))                # NaN preserved (Player2's y in the fixture)
        # kick off (home) -> home control from frame 0; pass by away at frame 9 (1-based 10) switches to away
        self.assertEqual(int(gt[2]), 0)
        self.assertEqual(int(gt[10]), 1)
        # ball out at 1-based 15 -> dead from index 14 until the throw-in at index 17
        self.assertTrue(all(int(gt[i]) == 2 for i in range(14, 17)))
        self.assertEqual(int(gt[18]), 0)                              # throw-in by home: home possession
        self.assertTrue(set(np.unique(gt)) <= {0, 1, 2, 3})

    def test_holdout_is_never_available_for_development(self):
        from src.external.registry import HOLDOUT_IDS, available, load
        names = available()
        for m in HOLDOUT_IDS:
            self.assertNotIn("skillcorner_{}".format(m), names)
            with self.assertRaises(KeyError):
                load("skillcorner_{}".format(m))

    def test_registry_only_lists_present_files(self):
        names = available()
        self.assertIsInstance(names, list)
        with self.assertRaises(KeyError):
            load("no_such_game")


if __name__ == "__main__":
    unittest.main()
