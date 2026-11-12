"""Compare correctly scaled gradient and Newton intercept updates inside skglm.

Both strategies use AndersonCD at the same pinned revision, with its native
Lipschitz coefficient steps, working sets, acceleration, and stopping rules.
Only LogisticGradient.intercept_update_step differs from the Newton datafit.

For each strategy and problem, select the loosest native tolerance whose fit
meets an externally evaluated relative primal-dual gap of 1e-6. Then warm that
fit and time five cold-start repetitions, interleaving the strategies. Timing
excludes input loading, tolerance selection, warm-up, and certificate evaluation.
Every timed endpoint is certified again; native stopping residuals do not define
success. Failed configurations are recorded and excluded from runtime ratios.

Run through run-skglm-controlled.sh to select the pinned source tree. A small
subset can be run directly with --cell-ids and --repeats.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy

from skglm import GeneralizedLinearEstimator
from skglm.datafits import Logistic
from skglm.penalties import L1
from skglm.solvers import AndersonCD


COMMIT = "8f9cbd77"
TARGET = 1.0e-6
TOL_GRID = 10.0 ** -np.arange(3.0, 14.0, 0.5)
MAX_ITER = 5000
MAX_EPOCHS = 200_000
STRATEGIES = ("gradient", "newton")


class LogisticGradient(Logistic):
    """Logistic datafit with the sharp global-curvature intercept step."""

    def intercept_update_step(self, y, Xw):
        # raw_grad is normalized by n, so g_0 / L_0 = 4 * sum(raw_grad).
        return 4.0 * np.sum(self.raw_grad(y, Xw))


def logistic_dual_point(eta, y01):
    """Enforce the intercept constraint without leaving the conjugate domain.

    This is the same centered-residual and intercept-only-anchor construction
    used by Intercepts._dual_point in src/loss.jl. Feature feasibility is imposed
    separately when evaluating the certificate.
    """
    residual = expit(eta) - y01
    residual -= residual.mean()
    ybar = y01.mean()
    direction = y01 + residual - ybar
    alpha = 1.0
    negative = direction < 0
    positive = direction > 0
    if negative.any():
        alpha = min(alpha, np.min(ybar / -direction[negative]))
    if positive.any():
        alpha = min(alpha, np.min((1.0 - ybar) / direction[positive]))
    alpha = np.clip(alpha, 0.0, 1.0)
    if 0.0 < alpha < 1.0:
        alpha *= 1.0 - np.sqrt(np.finfo(float).eps)
    anchor = ybar - y01
    return anchor + alpha * (residual - anchor)


def logistic_certificate(X, y_pm, lam, intercept, beta):
    """Evaluate the averaged logistic-lasso objective and a feasible dual bound."""
    eta = intercept + X @ beta
    primal = np.logaddexp(0.0, -y_pm * eta).mean() + lam * np.abs(beta).sum()
    y01 = (y_pm + 1.0) / 2.0
    theta = logistic_dual_point(eta, y01)
    feature_norm = np.max(np.abs(X.T @ theta)) / len(y01)
    theta /= max(1.0, feature_norm / lam) * (1.0 + 64 * np.finfo(float).eps)
    means = y01 + theta
    margin = 64 * np.finfo(float).eps
    if np.any(means < -margin) or np.any(means > 1.0 + margin):
        raise ValueError("logistic dual point is outside its conjugate domain")
    means = np.clip(means, 0.0, 1.0)
    dual = -np.mean(xlogy(means, means) + xlogy(1.0 - means, 1.0 - means))
    gap = primal - dual
    if not np.isfinite(gap) or gap < -margin * max(1.0, abs(primal)):
        raise ValueError(f"invalid primal-dual certificate: P={primal}, D={dual}")
    return dict(
        primal=float(primal),
        dual_bound=float(dual),
        relative_gap=float(max(0.0, gap) / max(abs(primal), 1.0e-15)),
    )


def fit(X, y_pm, lam, strategy, tol):
    """Fit from zero using skglm's unchanged AndersonCD implementation."""
    solver = AndersonCD(
        tol=tol,
        max_iter=MAX_ITER,
        max_epochs=MAX_EPOCHS,
        fit_intercept=True,
        warm_start=False,
        verbose=0,
    )
    datafit = LogisticGradient() if strategy == "gradient" else Logistic()
    estimator = GeneralizedLinearEstimator(
        datafit=datafit, penalty=L1(alpha=lam), solver=solver
    )
    start = time.perf_counter()
    estimator.fit(X, y_pm)
    runtime = time.perf_counter() - start
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    certificate = logistic_certificate(
        X, y_pm, lam, intercept, estimator.coef_.ravel()
    )
    return dict(
        **certificate,
        n_iter=int(estimator.n_iter_),
        stop_crit=float(estimator.stop_crit_),
        runtime=runtime,
        intercept=intercept,
    )


def run_cell(cells_dir, row, target, repeats):
    cell_path = cells_dir / row["cell_name"]
    X = np.asfortranarray(pd.read_csv(cell_path / "X.csv", header=None).to_numpy())
    y_pm = pd.read_csv(cell_path / "y.csv")["y_pm"].to_numpy(dtype=float)
    lam = json.loads((cell_path / "meta.json").read_text())["lambda"]
    common = dict(
        commit=COMMIT,
        **{
            key: row[key]
            for key in ("cell_id", "cell_name", "mu0", "reg", "seed", "n", "p")
        },
        lambda_=lam,
        target=target,
    )

    selected = {}
    for strategy in STRATEGIES:
        for tol in TOL_GRID:
            result = fit(X, y_pm, lam, strategy, tol)
            if result["relative_gap"] <= target:
                selected[strategy] = tol
                break
        else:
            yield dict(
                **common,
                strategy=strategy,
                tol=tol,
                repeat=0,
                **result,
                converged=False,
            )
        if strategy in selected:
            # Explicit warm-up at the selected tolerance, still from zero.
            fit(X, y_pm, lam, strategy, selected[strategy])

    rng = np.random.default_rng(int(row["cell_id"]))
    for repeat in range(1, repeats + 1):
        for strategy in rng.permutation(list(selected)):
            tol = selected[strategy]
            result = fit(X, y_pm, lam, strategy, tol)
            if result["relative_gap"] > target:
                raise RuntimeError(
                    f"{row['cell_name']} {strategy} missed the certified target"
                )
            yield dict(
                **common,
                strategy=strategy,
                tol=tol,
                repeat=repeat,
                **result,
                converged=True,
            )


COLS = [
    "commit",
    "strategy",
    "cell_id",
    "cell_name",
    "mu0",
    "reg",
    "seed",
    "n",
    "p",
    "lambda_",
    "target",
    "tol",
    "repeat",
    "n_iter",
    "stop_crit",
    "primal",
    "dual_bound",
    "relative_gap",
    "runtime",
    "converged",
    "intercept",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "skglm-controlled"
        / "index.csv",
    )
    parser.add_argument("--target", type=float, default=TARGET)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cell-ids", type=int, nargs="+")
    args = parser.parse_args()
    if args.repeats < 1 or args.target <= 0:
        parser.error("--repeats and --target must be positive")
    index = pd.read_csv(args.index)
    if args.cell_ids is not None:
        index = index[index.cell_id.isin(args.cell_ids)]
    if index.empty:
        parser.error("no problem cells selected")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"skglm AndersonCD @ {COMMIT}: gradient vs Newton; target={args.target:.0e}",
        flush=True,
    )
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS, lineterminator="\n")
        writer.writeheader()
        for _, row in index.iterrows():
            for result in run_cell(
                args.index.parent / "cells", row, args.target, args.repeats
            ):
                writer.writerow(result)
                stream.flush()
                status = "OK" if result["converged"] else "UNCERTIFIED"
                print(
                    f"  {row['cell_name']} {result['strategy']:8s} repeat={result['repeat']} "
                    f"tol={result['tol']:.1e} gap={result['relative_gap']:.2e} "
                    f"t={result['runtime']:.3f}s {status}",
                    flush=True,
                )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
