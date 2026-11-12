"""Tests for the discriminative frame classifier. Synthetic data only."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.degrade import calibrated_params, degrade  # noqa: E402
from src.external.synthetic import make_match  # noqa: E402
from src.model.frame_clf import (FEATURE_NAMES, build_features, decode_softmax, feature_matrix, fit_softmax,  # noqa: E402
                                 predict_log_proba, training_set)
from src.model.frame_features import build_frame_table  # noqa: E402
from src.model.hmm import extract_evidence, fit  # noqa: E402


def _case():
    tr, gt = make_match(0, minutes=8)
    return degrade(tr, 1, calibrated_params()), gt


class SoftmaxTests(unittest.TestCase):
    def test_learns_separable_classes(self):
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(m, 0.5, (300, 4)) for m in (-3, 0, 3)])
        y = np.repeat([0, 1, 2], 300)
        model = fit_softmax(X, y)
        pred = predict_log_proba(model, X).argmax(axis=1)
        self.assertGreater((pred == y).mean(), 0.97)
        np.testing.assert_allclose(np.exp(predict_log_proba(model, X)).sum(axis=1), 1.0)

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 5))
        y = rng.integers(0, 3, 500)
        a, b = fit_softmax(X, y), fit_softmax(X, y)
        np.testing.assert_array_equal(a.W, b.W)


class FeatureTests(unittest.TestCase):
    def test_shape_and_swap_symmetry(self):
        tr, _ = _case()
        X, Xs = build_features(tr, [(0, tr.n_frames)], swap_too=True)
        self.assertEqual(X.shape, (tr.n_frames, len(FEATURE_NAMES)))
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        np.testing.assert_allclose(X[:, idx["dmin_h"]], Xs[:, idx["dmin_a"]])
        np.testing.assert_allclose(X[:, idx["near5_a"]], Xs[:, idx["near5_h"]])
        np.testing.assert_allclose(X[:, idx["dpair"]], -Xs[:, idx["dpair"]], atol=1e-6)
        np.testing.assert_allclose(X[:, idx["crowd_log"]], Xs[:, idx["crowd_log"]])   # symmetric feature unchanged
        self.assertFalse(np.isnan(X).any())

    def test_hidden_ball_zeroes_ball_features_and_flags_it(self):
        tr, _ = _case()
        tr.ball[:] = np.nan
        X, _ = build_features(tr, [(0, tr.n_frames)])
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        self.assertEqual(float(X[:, idx["ball_vis"]].max()), 0.0)
        self.assertEqual(float(np.abs(X[:, idx["dmin_h"]]).max()), 0.0)


class PipelineTests(unittest.TestCase):
    def test_training_set_adds_swapped_copy_with_swapped_labels(self):
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        Xs = X + 100
        gt = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2])
        Xt, yt = training_set([X], [Xs], [gt], stride=1)
        self.assertEqual(len(yt), 2 * int((gt < 3).sum()))
        half = len(yt) // 2
        np.testing.assert_array_equal(yt[half:], np.array([1, 0, 2])[yt[:half]])

    def test_decode_outputs_valid_labels_inside_runs_only(self):
        tr, gt = _case()
        runs = [(100, tr.n_frames // 2), (tr.n_frames // 2 + 50, tr.n_frames - 100)]
        X, Xs = build_features(tr, runs, swap_too=True)
        Xt, yt = training_set([X], [Xs], [gt], stride=5)
        model = fit_softmax(Xt, yt, iters=60)
        nb = fit(extract_evidence(tr), gt, gt != 3)
        out = decode_softmax(model, X, nb.log_trans, nb.log_prior, runs, temp=0.3)
        self.assertEqual(set(np.unique(out[:100])), {3})
        self.assertTrue(set(np.unique(out[runs[0][0]:runs[0][1]])) <= {0, 1, 2})
        again = decode_softmax(model, X, nb.log_trans, nb.log_prior, runs, temp=0.3)
        np.testing.assert_array_equal(out, again)


class ProductionPathTests(unittest.TestCase):
    def test_softmax_params_roundtrip_and_feature_guard(self):
        import json
        import tempfile
        from src.model.io import load_softmax, save_softmax
        rng = np.random.default_rng(0)
        model = fit_softmax(rng.normal(size=(300, len(FEATURE_NAMES))), rng.integers(0, 3, 300), iters=20)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            save_softmax(path, model, {"x": 1})
            loaded, meta = load_softmax(path)
            np.testing.assert_allclose(loaded.W, model.W)
            np.testing.assert_allclose(loaded.log_prior, model.log_prior)
            payload = json.loads(path.read_text())
            payload["feature_names"] = payload["feature_names"][:-1]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_softmax(path)

    def test_predict_with_softmax_forces_dead_block_and_is_deterministic(self):
        from src.run import predict
        tr, gt = _case()
        n = tr.n_frames
        runs = [(100, n // 2), (n // 2 + 50, n - 100)]
        X, Xs = build_features(tr, runs, swap_too=True)
        Xt, yt = training_set([X], [Xs], [gt], stride=5)
        model = fit_softmax(Xt, yt, iters=40)
        nb = fit(extract_evidence(tr), gt, gt != 3)
        forced = [(runs[1][0], runs[1][0] + 200)]
        a = predict(tr, nb, runs, forced, softmax=model)
        b = predict(tr, nb, runs, forced, softmax=model)
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a[forced[0][0]:forced[0][1]] == 2).all())
        self.assertTrue((a[:100] == 3).all())
        nb_codes = predict(tr, nb, runs, forced)                   # the naive-Bayes path still works
        self.assertEqual(len(nb_codes), n)


if __name__ == "__main__":
    unittest.main()
