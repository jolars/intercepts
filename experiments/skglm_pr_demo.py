"""Demo: convergence of AndersonCD's intercept update on imbalanced logistic.

Run this script twice, once on each commit, to compare:

    git -C ~/projects/skglm checkout b03644fe -- skglm    # pre-fix
    python skglm_pr_demo.py
    git -C ~/projects/skglm checkout 8f9cbd77 -- skglm    # post-fix
    python skglm_pr_demo.py

Each run prints a summary table and writes a convergence plot named
``skglm_pr_demo_<gradient|newton>_<sha>.png`` next to this script.

The trajectories are captured by setting ``verbose=1`` on AndersonCD and
parsing its per-iteration ``Stopping criterion max violation`` line.
"""

import contextlib
import io
import os
import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Logistic
from skglm.penalties import L1
from skglm.solvers import AndersonCD

REPO = "/home/jola/projects/skglm"
HERE = os.path.dirname(os.path.abspath(__file__))
STOP_LINE = re.compile(r"Stopping criterion max violation:\s*([0-9.eE+-]+)")


def make_problem(mu_0, seed=0, n=500, p=200):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.where(rng.random(n) < mu_0, 1.0, -1.0)
    return X, y


def run_with_trace(X, y, alpha, tol, max_iter):
    est = GeneralizedLinearEstimator(
        datafit=Logistic(),
        penalty=L1(alpha),
        solver=AndersonCD(
            tol=tol,
            fit_intercept=True,
            max_iter=max_iter,
            warm_start=False,
            verbose=1,
        ),
    )
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        est.fit(X, y)
    elapsed = time.perf_counter() - t0

    trace = [float(m.group(1)) for m in STOP_LINE.finditer(buf.getvalue())]
    return {
        "trace": np.array(trace),
        "n_iter": int(est.n_iter_),
        "stop_crit": float(est.stop_crit_),
        "intercept": float(est.intercept_),
        "elapsed_s": elapsed,
        "converged": est.stop_crit_ < tol,
    }


def git_describe():
    sha = subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    subject = subprocess.check_output(
        ["git", "-C", REPO, "log", "-1", "--pretty=%s"], text=True
    ).strip()
    # Detect which strategy the working tree currently uses by checking
    # whether Logistic.intercept_update_step calls raw_hessian.
    with open(os.path.join(REPO, "skglm/datafits/single_task.py")) as f:
        src = f.read()
    parts = (
        src.split("class Logistic", 1)[1]
        .split("def intercept_update_step", 1)[1]
        .split("def ", 1)[0]
    )
    tag = "newton" if "raw_hessian" in parts else "gradient"
    return sha, subject, tag


def main():
    alpha = 0.05
    tol = 1e-8
    max_iter = 200
    mu_grid = (0.5, 0.8, 0.95, 0.97, 0.99)

    sha, subject, tag = git_describe()
    print("=" * 78)
    print(f"skglm @ {sha}  ({subject})")
    print(f"working tree intercept strategy: {tag}")
    print(f"alpha={alpha}, tol={tol:.0e}, max_iter={max_iter}, AndersonCD")
    print("=" * 78)
    print(
        f"{'mu_0':>6}  {'iters':>5}  {'stop_crit':>11}  "
        f"{'intercept':>10}  {'time(ms)':>9}  status"
    )
    print("-" * 78)

    results = {}
    for mu_0 in mu_grid:
        X, y = make_problem(mu_0)
        actual = float((y == 1).mean())
        res = run_with_trace(X, y, alpha, tol, max_iter)
        results[mu_0] = (actual, res)
        status = "OK" if res["converged"] else "STALL"
        print(
            f"{actual:>6.3f}  {res['n_iter']:>5}  {res['stop_crit']:>11.2e}  "
            f"{res['intercept']:>+10.4f}  {res['elapsed_s'] * 1000:>9.1f}  {status}"
        )

    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(mu_grid)))
    for color, (_, (actual, res)) in zip(colors, results.items()):
        trace = res["trace"]
        if trace.size == 0:
            continue
        ax.plot(
            np.arange(1, trace.size + 1),
            trace,
            color=color,
            label=rf"$\mu_0 = {actual:.3f}$",
        )
    ax.axhline(tol, ls="--", color="k", lw=0.8, label=f"tol = {tol:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("stop criterion (max KKT violation)")
    ax.set_title(f"AndersonCD on L1-logistic — intercept update: {tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    out = os.path.join(HERE, f"skglm_pr_demo_{tag}_{sha}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
