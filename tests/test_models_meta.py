"""The committed model files must have been fitted on EXTERNAL games only (RULES.md: no hand labels or data about the match)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.registry import METRICA_CSV_GAMES, SKILLCORNER_IDS  # noqa: E402

EXTERNAL = {"metrica_game3", "metrica_sample_game_3", "skillcorner_1874553"} \
    | {"metrica_game{}".format(g) for g in METRICA_CSV_GAMES} | {"skillcorner_{}".format(m) for m in SKILLCORNER_IDS}


class ModelProvenanceTests(unittest.TestCase):
    def _meta(self, name):
        path = ROOT / "models" / name
        self.assertTrue(path.exists(), name)
        return json.loads(path.read_text())["meta"]

    def test_holdout_games_never_seen_by_any_model(self):
        from src.external.registry import HOLDOUT_IDS
        for name in ("hmm_params.json", "softmax_params.json", "mlp_params.json"):
            text = json.dumps(self._meta(name))
            for m in HOLDOUT_IDS:
                self.assertNotIn(m, text, "{} mentions hold-out match {}".format(name, m))

    def test_only_external_games_in_meta(self):
        for name in ("hmm_params.json", "softmax_params.json", "mlp_params.json"):
            meta = self._meta(name)
            games = set(meta["n_frames"])
            self.assertTrue(games, name)
            self.assertTrue(games <= EXTERNAL, "{} lists non-external games: {}".format(name, games - EXTERNAL))
            text = json.dumps(meta).lower()
            for forbidden in ("bar-ath", "barcelona", "athletic", "competition", "tactical", "ground truth"):
                self.assertNotIn(forbidden, text, "{} meta mentions {!r}".format(name, forbidden))

    def test_frame_count_is_consistent_with_the_report(self):
        for name in ("softmax_params.json", "mlp_params.json"):
            meta = self._meta(name)
            self.assertEqual(sum(meta["n_frames"].values()), 1032904)
            self.assertEqual(len(meta["n_frames"]), 7)


if __name__ == "__main__":
    unittest.main()
