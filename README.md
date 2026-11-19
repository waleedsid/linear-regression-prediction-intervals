# Conformal Regression Lab

A small, runnable project for estimating uncertainty around linear regression
predictions using a held-out calibration sample.

## Run

Use Python 3.10 or later from this repository directory:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python conformal_regression.py --seed 42 --output results/example.json
```

## Problem and method

Point predictions alone leave unanswered how far a future outcome could be
from the estimate. This project fits ordinary least squares with an intercept
using NumPy's least-squares solver, then calibrates a constant-width prediction
interval using absolute residuals from separate observations.

For n calibration residuals and miscoverage alpha, use the residual at sorted
rank `ceil((n + 1) * (1 - alpha))`. A rank beyond n gives an infinite interval,
which explicitly handles calibration sets too small for a finite bound.
The method follows the split-conformal construction described by
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

The example creates 1,500 synthetic independent observations with three Gaussian
features and the response `4 + 2.5*x1 - 1.2*x2 + 0.7*x3 + noise`, where noise is
Gaussian with standard deviation 1.5. A seeded shuffle allocates 750 rows for
training, 375 for calibration and 375 for a final test. The mean-only baseline
uses only training targets. Neither calibration nor test targets fit coefficients.

## Measured example results

The included seed-42 run produced:

| Metric | Value |
| --- | ---: |
| Regression test RMSE | 1.6003 |
| Training-mean baseline test RMSE | 3.2107 |
| Target marginal coverage | 90% |
| Observed coverage on 375 test rows | 88.27% |
| Mean prediction interval width | 5.0857 |

Full precision results are in `results/example.json`. These are measurements on
synthetic data. Empirical coverage fluctuates on finite test samples.

## Assumptions and limitations

The marginal coverage guarantee depends on exchangeable calibration and future
observations and a model fitted independently of calibration outcomes. Coverage
for a particular feature value or subgroup may differ. Distribution shift and
time dependence need methods designed for those settings. Constant-width bounds
may be inefficient with heteroscedastic noise. Callers must enforce disjoint
training and calibration samples; the array interface cannot establish provenance.
Use calibration only for interval construction, and reserve a separate set for
model selection. Re-fitting clears the previous calibration.

## Project contents

- `conformal_regression.py` contains the estimator, quantile calculation and CLI.
- `tests/test_regression.py` covers analytic regression, exact quantile ranks,
  calibration isolation, refitting, invalid input and rank-deficient designs.
- `results/example.json` records the reproducible synthetic experiment.

All seven tests passed in the local validation run. Numerical results can vary
slightly across NumPy and linear algebra library versions.
