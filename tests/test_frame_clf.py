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


class MlpTests(unittest.TestCase):
    def test_learns_xor_that_a_linear_softmax_cannot(self):
        from src.model.frame_clf import fit_mlp
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)         # XOR of signs: not linearly separable
        lin = predict_log_proba(fit_softmax(X, y), X).argmax(axis=1)
        mlp = predict_log_proba(fit_mlp(X, y, hidden=16, iters=400), X).argmax(axis=1)
        self.assertLess((lin == y).mean(), 0.7)
        self.assertGreater((mlp == y).mean(), 0.9)

    def test_deterministic_and_proper_probabilities(self):
        from src.model.frame_clf import fit_mlp
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(400, 6)), rng.integers(0, 3, 400)
        a, b = fit_mlp(X, y, iters=30), fit_mlp(X, y, iters=30)
        np.testing.assert_array_equal(a.W1, b.W1)
        np.testing.assert_allclose(np.exp(predict_log_proba(a, X)).sum(axis=1), 1.0)


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


class ContextTests(unittest.TestCase):
    def test_shape_symmetry_and_no_nan(self):
        from src.model.frame_clf import CTX_NAMES, feature_names
        tr, _ = _case()
        X, Xs = build_features(tr, [(0, tr.n_frames)], swap_too=True, context=True)
        self.assertEqual(X.shape, (tr.n_frames, len(feature_names(True))))
        self.assertFalse(np.isnan(X).any())
        names = feature_names(True)
        idx = {n: i for i, n in enumerate(names)}
        for antisym in ("dpair_w2", "snear_w5", "gap_last_s", "gap_next_s", "s_smooth"):
            np.testing.assert_allclose(X[:, idx[antisym]], -Xs[:, idx[antisym]], atol=1e-5, err_msg=antisym)
        for sym in ("crowd_w2", "vis_w5", "gap_last_age", "out_w2"):
            np.testing.assert_allclose(X[:, idx[sym]], Xs[:, idx[sym]], atol=1e-5, err_msg=sym)
        self.assertEqual(len(CTX_NAMES), 19)

    def test_gap_carries_last_and_next_evidence(self):
        from src.model.frame_clf import context_matrix
        n, d = 400, len(FEATURE_NAMES)
        idx = {nm: i for i, nm in enumerate(FEATURE_NAMES)}
        X = np.zeros((n, d), dtype=np.float32)
        X[:, idx["ball_vis"]] = 1
        X[:, idx["team_ok"]] = 1
        X[:, idx["dmin_h"]] = np.log1p(1.0)         # home player 1 m from the ball ...
        X[:, idx["dmin_a"]] = np.log1p(30.0)        # ... away far: home nearer -> s > 0
        X[100:300, idx["ball_vis"]] = 0             # a 200-frame gap
        X[100:300, idx["dmin_h"]] = X[100:300, idx["dmin_a"]] = 0
        C = context_matrix(X)
        from src.model.frame_clf import CTX_NAMES
        ci = {nm: i for i, nm in enumerate(CTX_NAMES)}
        mid = 200
        self.assertGreater(C[mid, ci["gap_last_s"]], 0.9)
        self.assertGreater(C[mid, ci["gap_next_s"]], 0.9)
        # the +-1 s smoothing keeps the last reference alive 25 frames into the gap (frame 124), so the age at 200 is 76
        self.assertAlmostEqual(float(C[mid, ci["gap_last_age"]]), float(np.log1p(76 / 25.0)), places=3)
        self.assertEqual(float(C[50, ci["gap_last_s"]]), 0.0)         # outside the gap the gap features are 0
        self.assertGreater(C[50, ci["s_smooth"]], 0.9)


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

    def test_mlp_params_roundtrip_and_predict_path(self):
        import tempfile
        from src.model.frame_clf import fit_mlp
        from src.model.io import load_mlp, save_mlp
        from src.run import predict
        tr, gt = _case()
        n = tr.n_frames
        runs = [(100, n // 2), (n // 2 + 50, n - 100)]
        X, Xs = build_features(tr, runs, swap_too=True)
        Xt, yt = training_set([X], [Xs], [gt], stride=5)
        mlp = fit_mlp(Xt, yt, iters=30)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            save_mlp(path, mlp, {"n_frames": {"x": 1}})
            loaded, meta = load_mlp(path)
        np.testing.assert_allclose(loaded.W1, mlp.W1)
        np.testing.assert_allclose(predict_log_proba(loaded, X[:50]), predict_log_proba(mlp, X[:50]))
        nb = fit(extract_evidence(tr), gt, gt != 3)
        forced = [(runs[1][0], runs[1][0] + 100)]
        a = predict(tr, nb, runs, forced, softmax=loaded)
        self.assertTrue((a[forced[0][0]:forced[0][1]] == 2).all())
        self.assertTrue(set(np.unique(a[runs[0][0]:runs[0][1]])) <= {0, 1, 2})

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
