"""Linear regression with split-conformal prediction intervals."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _features(values):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError('Features must be a nonempty two-dimensional array.')
    if not np.isfinite(matrix).all():
        raise ValueError('Features must be finite.')
    return matrix


def _targets(values, rows):
    targets = np.asarray(values, dtype=float)
    if targets.shape != (rows,) or not np.isfinite(targets).all():
        raise ValueError('Targets must be a finite vector matching the feature rows.')
    return targets


def conformal_radius(residuals, alpha):
    """Return the finite-sample absolute-residual quantile, without interpolation."""
    scores = np.asarray(residuals, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError('Calibration scores must be a nonempty vector.')
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Calibration scores must be finite and nonnegative.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        return float('inf')
    return float(np.partition(scores, rank - 1)[rank - 1])


class ConformalLinearRegressor:
    """Fit OLS on training data and calibrate width on a separate sample."""

    def __init__(self, alpha=0.1):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be strictly between zero and one.')
        self.alpha = alpha
        self.coefficients = None
        self.radius = None

    def fit(self, features, targets):
        """Fit intercept and coefficients using least squares, resetting calibration."""
        features = _features(features)
        targets = _targets(targets, len(features))
        design = np.column_stack((np.ones(len(features)), features))
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        self.coefficients = coefficients
        self.radius = None
        return self

    def predict(self, features):
        """Predict point estimates for rows with the fitted feature count."""
        if self.coefficients is None:
            raise ValueError('Fit the model before prediction.')
        features = _features(features)
        if features.shape[1] != len(self.coefficients) - 1:
            raise ValueError('Feature count differs from training.')
        return self.coefficients[0] + features @ self.coefficients[1:]

    def calibrate(self, features, targets):
        """Calibrate using held-out observations never used to fit the model."""
        predictions = self.predict(features)
        targets = _targets(targets, len(predictions))
        self.radius = conformal_radius(np.abs(targets - predictions), self.alpha)
        return self

    def predict_interval(self, features):
        """Return point estimates and lower/upper prediction bounds."""
        if self.radius is None:
            raise ValueError('Calibrate the model before requesting intervals.')
        predictions = self.predict(features)
        return predictions, predictions - self.radius, predictions + self.radius


def run_example(seed=42):
    """Evaluate on a synthetic, independent test sample with known data generation."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(1500, 3))
    targets = 4 + features @ np.array([2.5, -1.2, 0.7]) + rng.normal(0, 1.5, 1500)
    order = rng.permutation(len(features))
    train, calibration, test = order[:750], order[750:1125], order[1125:]
    model = ConformalLinearRegressor(alpha=0.1).fit(features[train], targets[train])
    model.calibrate(features[calibration], targets[calibration])
    predictions, lower, upper = model.predict_interval(features[test])
    truth = targets[test]
    baseline = np.full(len(test), targets[train].mean())
    return {
        'seed': seed,
        'data': 'Synthetic independent Gaussian features and noise; no external dataset',
        'split_sizes': {'train': len(train), 'calibration': len(calibration), 'test': len(test)},
        'target_marginal_coverage': 1 - model.alpha,
        'test_coverage': float(np.mean((truth >= lower) & (truth <= upper))),
        'mean_interval_width': float(np.mean(upper - lower)),
        'test_rmse': float(np.sqrt(np.mean((truth - predictions) ** 2))),
        'mean_baseline_rmse': float(np.sqrt(np.mean((truth - baseline) ** 2))),
        'fitted_coefficients_including_intercept': model.coefficients.tolist(),
    }


def main():
    """Run the reproducible example and optionally save its JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(run_example(args.seed), indent=2, allow_nan=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
