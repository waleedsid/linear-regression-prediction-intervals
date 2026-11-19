"""Numerical and state-transition tests for conformal regression."""
import unittest
import numpy as np
from conformal_regression import ConformalLinearRegressor, conformal_radius, run_example


class RegressionTests(unittest.TestCase):
    def test_recovers_exact_linear_function(self):
        x = np.arange(10.0).reshape(-1, 1)
        model = ConformalLinearRegressor().fit(x, 3 + 2 * x[:, 0])
        np.testing.assert_allclose(model.predict([[11], [12]]), [25, 27])

    def test_finite_sample_rank(self):
        self.assertEqual(conformal_radius([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2), 8)
        self.assertTrue(np.isinf(conformal_radius([1, 2], 0.1)))

    def test_calibration_does_not_refit(self):
        model = ConformalLinearRegressor(0.5).fit([[0], [1]], [0, 1])
        before = model.coefficients.copy()
        model.calibrate([[2], [3], [4]], [3, 5, 7])
        np.testing.assert_array_equal(before, model.coefficients)
        point, lower, upper = model.predict_interval([[5]])
        np.testing.assert_allclose(point, [5])
        np.testing.assert_allclose(lower, [3])
        np.testing.assert_allclose(upper, [7])

    def test_refit_requires_fresh_calibration(self):
        model = ConformalLinearRegressor().fit([[0], [1]], [0, 1])
        model.calibrate([[2], [3]], [2, 3])
        model.fit([[0], [1]], [5, 6])
        with self.assertRaises(ValueError):
            model.predict_interval([[2]])

    def test_invalid_inputs(self):
        for alpha in (0, 1, float('nan')):
            with self.assertRaises(ValueError):
                ConformalLinearRegressor(alpha)
        for features, targets in (([], []), ([[float('nan')]], [1]), ([[1], [2]], [1])):
            with self.assertRaises(ValueError):
                ConformalLinearRegressor().fit(features, targets)
        model = ConformalLinearRegressor().fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[1, 2]])
        with self.assertRaises(ValueError):
            ConformalLinearRegressor().predict([[1]])

    def test_rank_deficient_design(self):
        model = ConformalLinearRegressor().fit([[0, 0], [1, 1], [2, 2]], [1, 3, 5])
        np.testing.assert_allclose(model.predict([[3, 3]]), [7])

    def test_example_reproducible_and_beats_baseline(self):
        first = run_example()
        self.assertEqual(first, run_example())
        self.assertLess(first['test_rmse'], first['mean_baseline_rmse'])
        self.assertGreater(first['test_coverage'], 0.80)
        self.assertLessEqual(first['test_coverage'], 1)


if __name__ == '__main__':
    unittest.main()
