# `results/skglm-controlled/`

Cached outputs of the within-skglm controlled comparison driven by
`experiments/run-skglm-controlled.sh`; consumed by `@fig-skglm-controlled`.

- `index.csv` — one row per problem cell: `(mu0, reg, seed)` grid points with
  the realised problem metadata (`n`, `p`, `lambda`, `lambda_max`, `ybar`,
  `n_pos`, `n_neg`) emitted by `sim-skglm-controlled-problem.jl`.
- `results.csv` — one row per `(commit, cell)` solve: skglm `AndersonCD`'s
  `n_iter`, `stop_crit`, `primal`, `runtime`, `converged`, and `intercept` at
  the pre-fix (`b03644fe`) and post-fix (`8f9cbd77`) commits of
  `/home/jola/projects/skglm`.

The per-cell scratch under `cells/` is `.gitignore`d.
