# `results/skglm-controlled/`

Cached outputs of the within-skglm controlled comparison driven by
`experiments/run-skglm-controlled.sh`; consumed by `@fig-skglm-controlled`.

- `index.csv` — one row per problem cell: `(mu0, reg, seed)` grid points with
  the realized problem metadata (`n`, `p`, `lambda`, `lambda_max`, `ybar`,
  `n_pos`, `n_neg`) emitted by `sim-skglm-controlled-problem.jl`.
- `results.csv` — one row per `(cell, strategy, repeat)` fit at the same skglm
  revision (`8f9cbd77`), supplied by the source tree pinned in `devenv.lock`.
  The gradient datafit subtracts `4*g0` (`L0 = 1/4` for the averaged logistic
  loss), and the native Newton datafit subtracts `g0/H00`. Both use the native
  `AndersonCD` coefficient updates, working sets, acceleration, and stopping
  logic. The cache records the selected native tolerance, primal, feasible dual
  bound, relative primal-dual gap, runtime, native iteration count, stopping
  residual, and fitted intercept.

The driver selects the loosest native tolerance on a half-decade grid from
`1e-3` to `10^-13.5` that reaches the externally evaluated relative primal-dual
gap target of `1e-6`. It then warms that fit and times five cold-start
repetitions, interleaving the strategies in a seeded random order. Timings
exclude loading, tolerance calibration, warm-up, and certificate evaluation.
Every timed fit is certified again. A configuration that misses the target is
recorded with `repeat = 0` and `converged = False`; it cannot contribute to a
runtime ratio. The figure takes each strategy's median runtime per problem,
averages these medians across the three seeds, and forms the gradient-to-Newton
ratio only when all three pairs are certified.

The per-cell scratch under `cells/` is `.gitignore`d. The wrapper checks every
indexed cell for nonempty `X.csv`, `y.csv`, and `meta.json` files and
regenerates the problem family if any are missing or empty. It checks the
generated inputs before running either strategy, and replaces `results.csv` only
after the comparison completes successfully.
