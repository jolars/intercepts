"""Check the scientific control and its external accuracy certificate.

Run with PYTHONPATH="$SKGLM_POST_FIX_SOURCE" so the Newton variant uses the
same pinned source tree as the reproduction wrapper.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAVE_SOLVER_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "pandas", "scipy", "skglm")
)
if HAVE_SOLVER_DEPS:
    import numpy as np
    from scipy.special import expit, xlogy

    SPEC = importlib.util.spec_from_file_location(
        "skglm_controlled", ROOT / "experiments" / "sim-skglm-controlled.py"
    )
    CONTROL = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CONTROL)


@unittest.skipUnless(HAVE_SOLVER_DEPS, "requires the Python experiment environment")
class SkglmControlTests(unittest.TestCase):
    def test_gradient_step_uses_sharp_global_curvature(self):
        y = np.array([-1.0, 1.0, 1.0, 1.0])
        eta = np.array([-0.7, 1.2, 2.0, -0.1])
        gradient = np.mean(-y * expit(-y * eta))
        self.assertAlmostEqual(
            CONTROL.LogisticGradient().intercept_update_step(y, eta), 4.0 * gradient
        )

    def test_centered_dual_residual_stays_in_domain(self):
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        eta = np.array([30.0, -30.0, 30.0, -30.0, 0.0])
        centered = expit(eta) - y
        centered -= centered.mean()
        self.assertGreater(np.max(y + centered), 1.0)
        theta = CONTROL.logistic_dual_point(eta, y)
        self.assertAlmostEqual(theta.sum(), 0.0)
        self.assertTrue(np.all(y + theta >= 0.0))
        self.assertTrue(np.all(y + theta <= 1.0))

    def test_intercept_only_optimum_has_zero_gap(self):
        y = np.r_[np.ones(95), -np.ones(5)]
        X = np.zeros((len(y), 3))
        intercept = np.log(0.95 / 0.05)
        expected = -xlogy(0.95, 0.95) - xlogy(0.05, 0.05)
        result = CONTROL.logistic_certificate(X, y, 0.1, intercept, np.zeros(3))
        self.assertAlmostEqual(result["primal"], expected)
        self.assertAlmostEqual(result["dual_bound"], expected)
        self.assertLess(result["relative_gap"], 1.0e-12)
        initial = CONTROL.logistic_certificate(X, y, 0.1, 0.0, np.zeros(3))
        self.assertGreater(initial["relative_gap"], 0.5)

    def test_feature_constraint_and_loss_normalization(self):
        # With X_i = y_i and balanced labels, beta*=log((1-lambda)/lambda),
        # beta_0*=0 and P*=binary_entropy(lambda), independently of n.
        for n in (20, 100):
            y = np.tile([-1.0, 1.0], n // 2)
            X = y[:, None]
            for lam in (0.001, 0.1, 0.45):
                expected = -xlogy(lam, lam) - xlogy(1.0 - lam, 1.0 - lam)
                beta = np.array([np.log((1.0 - lam) / lam)])
                result = CONTROL.logistic_certificate(X, y, lam, 0.0, beta)
                self.assertAlmostEqual(result["primal"], expected)
                self.assertAlmostEqual(result["dual_bound"], expected)
                self.assertLess(result["relative_gap"], 1.0e-10)
                for candidate in (np.zeros(1), 2.0 * beta):
                    result = CONTROL.logistic_certificate(X, y, lam, 0.0, candidate)
                    self.assertLessEqual(result["dual_bound"], expected + 1.0e-12)
                    self.assertGreaterEqual(result["primal"], expected - 1.0e-12)

    def test_both_skglm_variants_reach_certified_accuracy(self):
        rng = np.random.default_rng(42)
        X = np.asfortranarray(rng.normal(size=(100, 5)))
        y = np.r_[np.ones(95), -np.ones(5)]
        expected = -xlogy(0.95, 0.95) - xlogy(0.05, 0.05)
        for strategy in CONTROL.STRATEGIES:
            result = CONTROL.fit(X, y, 1.0, strategy, 1.0e-10)
            self.assertLessEqual(result["relative_gap"], CONTROL.TARGET)
            self.assertAlmostEqual(result["primal"], expected, places=8)
            self.assertAlmostEqual(result["intercept"], np.log(19.0), places=6)


if __name__ == "__main__":
    unittest.main()
