"""Fig D: LIBLINEAR (newGLMNET) and BlitzL1 on the shared imbalanced logistic
problem.

Both are prox-Newton with local weights (Bucket B in the classification table)
and additionally apply the convergence strategy to the intercept after each
prox-Newton subproblem. The paper predicts they converge fast --- comparable
to glmnet `Newton` and biglasso `Newton`. This experiment verifies that
prediction directly on the production codebases.

LIBLINEAR is driven through scikit-learn's `LogisticRegression(solver=
"liblinear")` wrapper. To approximate an unpenalized intercept we set
`intercept_scaling` large; the residual L1 penalty on the intercept becomes
negligible.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

import blitzl1
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# Quiet sklearn FutureWarning noise about penalty/l1_ratio deprecation.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sys.stdout.reconfigure(line_buffering=True)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "real-solvers"

X = pd.read_csv(OUTDIR / "X.csv", header=None).to_numpy()
ydf = pd.read_csv(OUTDIR / "y.csv")
y_pm = ydf["y_pm"].to_numpy().astype(float)
y01 = ydf["y01"].to_numpy().astype(int)
meta = json.loads((OUTDIR / "meta.json").read_text())
lam = meta["lambda"]
n = meta["n"]


def logistic_primal(beta0, beta):
    eta = beta0 + X @ beta
    z = -y_pm * eta
    loss = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    return loss.mean() + lam * np.abs(beta).sum()


# --- LIBLINEAR (scikit-learn wrapper, solver="liblinear") -----------------
# LIBLINEAR's L1-LR minimizes  ||w||_1 + C * sum_i log(1 + exp(-y_i x_i^T w))
# Match to our (1/n) * sum log + lambda * ||w||_1  =>  C = 1/(n * lambda).
liblinear_C = 1.0 / (n * lam)

# intercept_scaling >> 1 makes the implicit bias-feature large, so the
# L1 penalty on the corresponding coefficient becomes a negligible fraction
# of the loss term; the intercept is effectively unpenalized.
intercept_scaling = 1e6

# LIBLINEAR's `max_iter` is a hard cap on outer iterations; setting it too
# high lets very tight `tol` values spin for many minutes without improving
# the iterate. 5000 is plenty for this problem -- at tol=1e-8 LIBLINEAR
# already needs ~10000 iterations and is no longer making meaningful progress.
liblinear_tols = 10 ** np.arange(-1.0, -8.5, -0.5)
liblinear_rows = []
print("=== LIBLINEAR (newGLMNET via scikit-learn) ===")
for tol in liblinear_tols:
    clf = LogisticRegression(
        penalty="l1",
        C=liblinear_C,
        solver="liblinear",
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        tol=tol,
        max_iter=5000,
    )
    t0 = time.perf_counter()
    clf.fit(X, y01)
    runtime = time.perf_counter() - t0
    beta0 = float(clf.intercept_[0])
    beta = clf.coef_[0].copy()
    primal = logistic_primal(beta0, beta)
    n_iter = int(clf.n_iter_[0])
    print(
        f"  tol={tol:.0e}  n_iter={n_iter:5d}  a0={beta0:.4f}  "
        f"primal={primal:.10f}  time={runtime:.3f}s"
    )
    liblinear_rows.append(
        dict(solver="LIBLINEAR", tol=tol, n_iter=n_iter, primal=primal, runtime=runtime)
    )

# --- BlitzL1 ---------------------------------------------------------------
# BlitzL1's L1 penalty is on the raw L1 norm of the coefficient vector.
# Its `solve(l1_penalty, ...)` takes the penalty coefficient directly.
# Match: our objective is (1/n) sum_i log + lambda * ||w||_1, BlitzL1
# minimizes sum_i log + l1_penalty * ||w||_1, so l1_penalty = n * lambda.
blitz_l1 = n * lam

blitz_tols = 10 ** np.arange(-1.0, -10.5, -0.5)
blitz_rows = []
print("=== BlitzL1 ===")
blitzl1.set_use_intercept(True)
blitzl1.set_verbose(False)
prob = blitzl1.LogRegProblem(X, y_pm)
for tol in blitz_tols:
    blitzl1.set_tolerance(tol)
    t0 = time.perf_counter()
    sol = prob.solve(blitz_l1, max_iter=500, log_directory="")
    runtime = time.perf_counter() - t0
    beta = np.asarray(sol.x).ravel()
    beta0 = float(sol.intercept)
    primal = logistic_primal(beta0, beta)
    # BlitzL1 doesn't expose an iteration counter on the Solution object;
    # use runtime as the work axis for this solver.
    print(
        f"  tol={tol:.0e}  a0={beta0:.4f}  primal={primal:.10f}  time={runtime:.3f}s"
    )
    blitz_rows.append(
        dict(solver="BlitzL1", tol=tol, n_iter=-1, primal=primal, runtime=runtime)
    )

rows = liblinear_rows + blitz_rows
pd.DataFrame(rows).to_csv(OUTDIR / "proxnewton.csv", index=False)
print(f"Wrote {OUTDIR / 'proxnewton.csv'}")
