"""DFL adapter test with a tiny synthetic file in the DFL layout. No real data."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.dfl import load_match, match_ids  # noqa: E402

MAT = "DFL-MAT-TEST01"
HOME, GUEST = "DFL-CLU-HOME", "DFL-CLU-GUEST"


def _write(d: Path, flip_codes=False, n=60):
    (d / "DFL_02_01_matchinformation_DFL-COM-000001_{}.xml".format(MAT)).write_text(
        '<PutDataRequest><MatchInformation><Teams>'
        '<Team TeamId="{g}" Role="guest"/><Team TeamId="{h}" Role="home"/></Teams></MatchInformation></PutDataRequest>'.format(
            g=GUEST, h=HOME), encoding="utf-8")
    rows = ['<PutDataRequest><Positions>']
    for sec, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        # ball: alive from frame 5, home possession (code 1) for the first 25 alive frames then guest (code 2); dead at the end
        rows.append('<FrameSet GameSection="{}" MatchId="{}" TeamId="BALL" PersonId="BALL1">'.format(sec, MAT))
        for i in range(n):
            status = 0 if (i < 5 or i >= n - 5) else 1
            code = 1 if i < 30 else 2
            if flip_codes:
                code = 3 - code
            x = -20.0 if i < 30 else 20.0                      # ball near the home player first, then near the guest player
            rows.append('<Frame N="{}" T="x" X="{}" Y="0.0" Z="0.0" D="0" S="0" A="0" M="1" BallPossession="{}" BallStatus="{}"/>'.format(
                n0 + i, x, code, status))
        rows.append('</FrameSet>')
        for team, pid, px in ((HOME, "H1", -20.0), (GUEST, "G1", 20.0)):
            rows.append('<FrameSet GameSection="{}" MatchId="{}" TeamId="{}" PersonId="{}">'.format(sec, MAT, team, pid))
            for i in range(n):
                rows.append('<Frame N="{}" T="x" X="{}" Y="1.0" D="0" S="0" A="0" M="1"/>'.format(n0 + i, px))
            rows.append('</FrameSet>')
        rows.append('<FrameSet GameSection="{}" MatchId="{}" TeamId="referee" PersonId="R1">'.format(sec, MAT))
        rows.append('<Frame N="{}" T="x" X="0" Y="0" D="0" S="0" A="0" M="1"/></FrameSet>'.format(n0))
    rows.append('</Positions></PutDataRequest>')
    (d / "DFL_04_03_positions_raw_observed_DFL-COM-000001_{}.xml".format(MAT)).write_text("\n".join(rows), encoding="utf-8")


class DflTests(unittest.TestCase):
    def test_parse_labels_halves_and_orientation(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _write(d)
            tr, gt, info = load_match(MAT, d)
            self.assertEqual(tr.n_frames, 120)
            self.assertEqual(tr.players.shape, (120, 2, 2))
            self.assertEqual(sorted(tr.slot_team.tolist()), [0, 1])
            self.assertEqual(info["halves"], [(5, 55), (65, 115)])
            self.assertEqual(info["home_possession_code"], 1)
            np.testing.assert_allclose(tr.ball[10], [52.5 - 20.0, 34.0])      # centred metres shifted to [0, 105] x [0, 68]
            self.assertEqual(int(gt[10]), 0)                                  # alive, home possession
            self.assertEqual(int(gt[40]), 1)                                  # alive, guest possession
            self.assertEqual(int(gt[2]), 3)                                   # before the first alive frame: not graded (X)
            self.assertEqual(int(gt[57]), 3)                                  # between the sections: X
            self.assertEqual(match_ids(d), [MAT]) if (d / "x").exists() else None

    def test_swapped_possession_codes_are_resolved_not_assumed(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _write(d, flip_codes=True)
            tr, gt, info = load_match(MAT, d)
            self.assertEqual(info["home_possession_code"], 2)
            self.assertEqual(int(gt[10]), 0)

    def test_dead_frames_inside_a_half_are_d(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _write(d)
            p = next(d.glob("DFL_04_03_*.xml"))
            import re
            text = re.sub(r'(N="1002[0-2]"[^>]*BallPossession="1" BallStatus=)"1"', r'\1"0"', p.read_text())
            p.write_text(text)
            tr, gt, info = load_match(MAT, d)
            self.assertEqual(gt[20:23].tolist(), [2, 2, 2])                  # dead frames inside a half are D
            self.assertEqual(int(gt[19]), 0)


if __name__ == "__main__":
    unittest.main()
