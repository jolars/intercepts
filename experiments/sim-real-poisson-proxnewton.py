"""skglm ProxNewton on the congress109 Poisson problem.

ProxNewton is the production-grade prox-Newton solver: each outer pass
constructs the second-order Taylor quadratic at the current iterate
(local weights w_i = exp(eta_i) for Poisson) and minimises the
penalised quadratic via CD. The intercept is updated as part of the
inner subproblem. This is Bucket B (local-IRLS-equivalent) in the
paper's classification.

The logistic counterpart of this script runs LIBLINEAR and BlitzL1, but
neither supports Poisson regression -- LIBLINEAR is binary-logistic
only, BlitzL1 hard-codes the logistic datafit -- so the Poisson panel
uses skglm's ProxNewton as the prox-Newton representative. Output:
(tol, n_iter, primal, runtime) per tolerance.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from skglm import GeneralizedLinearEstimator
from skglm.datafits import Poisson
from skglm.penalties import L1
from skglm.solvers import ProxNewton

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)


ROOT = Path(__file__).resolve().parent.parent
PROBDIR = ROOT / "results" / "real-solvers-poisson" / "congress109"


tol_grid = sorted(
    set(np.round(10 ** np.arange(-1.0, -8.5, -0.25), 16)),
    reverse=True,
)


def run_sweep(probdir, problem):
    X = pd.read_csv(probdir / "X.csv", header=None).to_numpy()
    ydf = pd.read_csv(probdir / "y.csv")
    y = ydf["y"].to_numpy().astype(float)
    meta = json.loads((probdir / "meta.json").read_text())
    lam = meta["lambda"]

    def poisson_primal(beta0, beta):
        eta = beta0 + X @ beta
        return float(np.mean(np.exp(eta) - y * eta) + lam * np.abs(beta).sum())

    rows = []
    print(f"=== skglm ProxNewton Poisson problem={problem} ===")
    for tol in tol_grid:
        solver = ProxNewton(
            tol=tol,
            max_iter=500,
            fit_intercept=True,
            warm_start=False,
            verbose=0,
        )
        est = GeneralizedLinearEstimator(
            datafit=Poisson(),
            penalty=L1(alpha=lam),
            solver=solver,
        )

        t0 = time.perf_counter()
        try:
            est.fit(X, y)
            runtime = time.perf_counter() - t0
        except Exception as e:
            runtime = time.perf_counter() - t0
            print(f"  tol={tol:.0e}  FAILED: {e}")
            continue

        beta0 = (
            float(est.intercept_)
            if np.ndim(est.intercept_) == 0
            else float(est.intercept_[0])
        )
        beta = np.asarray(est.coef_).ravel()
        primal = poisson_primal(beta0, beta)
        n_iter = (
            int(est.n_iter_) if np.ndim(est.n_iter_) == 0 else int(est.n_iter_[0])
        )
        print(
            f"  tol={tol:.0e}  n_iter={n_iter:5d}  a0={beta0:.4f}  "
            f"primal={primal:.10f}  time={runtime:.3f}s"
        )
        rows.append(
            dict(
                problem=problem,
                solver="ProxNewton",
                tol=tol,
                n_iter=n_iter,
                primal=primal,
                runtime=runtime,
            )
        )
    return pd.DataFrame(rows)


out = run_sweep(PROBDIR, "congress109")
out.to_csv(PROBDIR / "proxnewton.csv", index=False)
print(f"Wrote {PROBDIR / 'proxnewton.csv'}")
