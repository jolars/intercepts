"""Fig C: skglm AndersonCD on the shared imbalanced logistic problem.

skglm 0.5 AndersonCD uses constant per-coordinate Lipschitz steps with the
global 1/4 upper bound on the logistic Hessian. The intercept is updated with
the fixed Lipschitz step ``mean(-y*sigmoid(-y*Xw))/4``, i.e. exactly the
gradient strategy with ``L_0 = n/4``. Lemma 3.3 predicts this stalls under
imbalance, which is what this figure documents.

skglm 0.6 (merged after these experiments were run, PR
https://github.com/scikit-learn-contrib/skglm/pull/337) replaces this with a
Newton intercept step; pin the package version to 0.5 to reproduce the figure.

Sweeps ``tol`` from loose to tight; records (n_iter, primal) at termination.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from skglm import GeneralizedLinearEstimator
from skglm.datafits import Logistic
from skglm.penalties import L1
from skglm.solvers import AndersonCD


ROOT = Path(__file__).resolve().parent.parent
REAL_SOLVERS = ROOT / "results" / "real-solvers"


# Tol grid trimmed at 10^-7 (well past the slow-ramp signature shown in
# fig-real-skglm; tighter tols just trace the Anderson-acceleration scatter
# near F*). max_iter capped at 2000 so the gradient-strategy stall on
# news20-3pct -- which by design has no tight-tol convergence -- bounds
# wall time instead of escalating doubling per tol step.
tol_grid = sorted(
    set(np.round(10 ** np.arange(-1.0, -7.25, -0.25), 16)),
    reverse=True,
)


def run_sweep(probdir, problem):
    X = pd.read_csv(probdir / "X.csv", header=None).to_numpy()
    ydf = pd.read_csv(probdir / "y.csv")
    y_pm = ydf["y_pm"].to_numpy().astype(float)
    meta = json.loads((probdir / "meta.json").read_text())
    lam = meta["lambda"]

    def logistic_primal(beta0, beta):
        eta = beta0 + X @ beta
        z = -y_pm * eta
        loss = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        return loss.mean() + lam * np.abs(beta).sum()

    rows = []
    print(f"=== skglm AndersonCD problem={problem} ===")
    for tol in tol_grid:
        solver = AndersonCD(
            tol=tol,
            max_iter=2000,
            max_epochs=200_000,
            fit_intercept=True,
            warm_start=False,
            verbose=0,
        )
        est = GeneralizedLinearEstimator(
            datafit=Logistic(),
            penalty=L1(alpha=lam),
            solver=solver,
        )

        t0 = time.perf_counter()
        est.fit(X, y_pm)
        runtime = time.perf_counter() - t0

        beta0 = (
            float(est.intercept_)
            if np.ndim(est.intercept_) == 0
            else float(est.intercept_[0])
        )
        beta = np.asarray(est.coef_).ravel()
        primal = logistic_primal(beta0, beta)
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
                tol=tol,
                n_iter=n_iter,
                primal=primal,
                runtime=runtime,
            )
        )
    return pd.DataFrame(rows)


synthetic_dir = REAL_SOLVERS
w1a_dir = REAL_SOLVERS / "w1a"
news20_dir = REAL_SOLVERS / "news20-3pct"

out_synth = run_sweep(synthetic_dir, "synthetic")
out_synth.to_csv(synthetic_dir / "skglm.csv", index=False)
print(f"Wrote {synthetic_dir / 'skglm.csv'}")

out_w1a = run_sweep(w1a_dir, "w1a")
out_w1a.to_csv(w1a_dir / "skglm.csv", index=False)
print(f"Wrote {w1a_dir / 'skglm.csv'}")

out_news20 = run_sweep(news20_dir, "news20-3pct")
out_news20.to_csv(news20_dir / "skglm.csv", index=False)
print(f"Wrote {news20_dir / 'skglm.csv'}")
