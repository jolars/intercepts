"""skglm AndersonCD on the congress109 Poisson problem.

The logistic version of this script pairs ``Logistic()`` datafit with skglm's
``AndersonCD`` solver, which is the Bucket A representative in the production
shootout (constant per-coordinate Lipschitz step on logistic's bounded
Hessian). The Poisson analogue is impossible by construction: ``AndersonCD``
calls ``datafit.get_lipschitz()`` during ``custom_checks``, and
``Poisson`` declines to implement it because exp(eta) has no global upper
bound on R. Raising the exception at attribute time is therefore exactly
the paper's claim, expressed as a runtime check: no Bucket A intercept
strategy exists for Poisson.

This script records that finding to ``skglm.csv`` as a single annotated row
so the figure code can detect it explicitly rather than silently rendering
an empty panel.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Poisson
from skglm.penalties import L1
from skglm.solvers import AndersonCD

sys.stdout.reconfigure(line_buffering=True)


ROOT = Path(__file__).resolve().parent.parent
PROBDIR = ROOT / "results" / "real-solvers-poisson" / "congress109"


def main():
    X = pd.read_csv(PROBDIR / "X.csv", header=None).to_numpy()
    ydf = pd.read_csv(PROBDIR / "y.csv")
    y = ydf["y"].to_numpy().astype(float)
    meta = json.loads((PROBDIR / "meta.json").read_text())
    lam = meta["lambda"]

    est = GeneralizedLinearEstimator(
        datafit=Poisson(),
        penalty=L1(alpha=lam),
        solver=AndersonCD(
            tol=1e-3,
            max_iter=10,
            fit_intercept=True,
            warm_start=False,
            verbose=0,
        ),
    )
    t0 = time.perf_counter()
    try:
        est.fit(X, y)
        runtime = time.perf_counter() - t0
        primal = float(
            np.mean(
                np.exp(est.intercept_ + X @ est.coef_)
                - y * (est.intercept_ + X @ est.coef_)
            )
            + lam * np.abs(est.coef_).sum()
        )
        row = dict(
            problem="congress109",
            tol=1e-3,
            n_iter=int(est.n_iter_),
            primal=primal,
            runtime=runtime,
            status="ran",
        )
        print(f"AndersonCD ran (unexpectedly): primal={primal:.6f}")
    except AttributeError as e:
        runtime = time.perf_counter() - t0
        msg = str(e).splitlines()[0]
        print(
            "skglm AndersonCD + Poisson is unsupported by construction:\n"
            f"  {msg}\n"
            "No Bucket A intercept strategy exists for Poisson (f''(eta) "
            "= exp(eta) has no global upper bound)."
        )
        row = dict(
            problem="congress109",
            tol=float("nan"),
            n_iter=-1,
            primal=float("nan"),
            runtime=runtime,
            status="unsupported_no_lipschitz",
        )

    pd.DataFrame([row]).to_csv(PROBDIR / "skglm.csv", index=False)
    print(f"Wrote {PROBDIR / 'skglm.csv'}")


if __name__ == "__main__":
    main()
