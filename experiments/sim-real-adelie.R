## Fig: adelie on the shared imbalanced logistic problem.
##
## adelie (Yang & Hastie 2024) is a direct-Newton / IRLS solver: it linearizes
## the non-Gaussian loss into a weighted Gaussian surrogate on each outer pass
## and calls an inner CD pin solver on that surrogate. The intercept is the
## implicit weighted-residual-mean of the centered-features parameterization,
## so on the surrogate it is at the optimum by construction. This puts adelie
## in the same bucket as glmnet `Newton` / biglasso `Newton` / Lasso.jl /
## ncvreg in the paper's classification (local-IRLS + convergence-on-quadratic
## intercept). The framework predicts fast convergence.
##
## Tolerance knob: adelie exposes both `tol` (inner CD) and `irls_tol` (outer
## IRLS). For a clean convergence-vs-cost sweep we set them equal. We also
## disable adelie's path-level early-exit so the single-lambda fit runs to
## convergence at every tolerance setting.

suppressPackageStartupMessages({
  library(adelie)
  library(jsonlite)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  flag <- "--file="
  match <- grep(flag, args)
  if (length(match) > 0) {
    normalizePath(dirname(sub(flag, "", args[match])))
  } else if (!is.null(sys.frame(1)$ofile)) {
    normalizePath(dirname(sys.frame(1)$ofile))
  } else {
    normalizePath(".")
  }
}
root <- normalizePath(file.path(get_script_dir(), ".."))
outdir <- file.path(root, "results", "real-solvers")

X <- as.matrix(read.csv(file.path(outdir, "X.csv"), header = FALSE))
ydf <- read.csv(file.path(outdir, "y.csv"))
y01 <- ydf$y01
y_pm <- ydf$y_pm
meta <- fromJSON(file.path(outdir, "meta.json"))
lambda <- meta$lambda

logistic_primal <- function(beta0, beta) {
  eta <- as.numeric(beta0 + X %*% beta)
  z <- -y_pm * eta
  loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  mean(loss) + lambda * sum(abs(beta))
}

## adelie's `tol` is dimensionless relative to the deviance gap (effective
## tolerance ~ tol * (loss_null - loss_full) / hess_sum inside the pin solver),
## so tol=3e-4 already drives suboptimality to ~7e-5 on this problem. Tighter
## than ~3e-4 the outer IRLS criterion `|Σ (resid - resid_prev)·(eta - eta_prev)|
## ≤ irls_tol` runs out of progress for a long time before the iterate cap,
## so each tol below the wall takes many minutes (and may fail outright by
## returning a state with zero stored λs). The working range gives a clean
## three-order-of-magnitude span in suboptimality.
tol_grid <- 10 ^ seq(-1, -3.5, by = -0.25)

rows <- list()
cat("=== adelie ===\n")
for (tol in tol_grid) {
  t0 <- proc.time()[["elapsed"]]
  state <- tryCatch(
    suppressWarnings(adelie::grpnet(
      X = X,
      glm = adelie::glm.binomial(y = y01),
      lambda = lambda,
      intercept = TRUE,
      standardize = FALSE,
      early_exit = FALSE,
      adev_tol = 1.0,
      tol = tol,
      irls_tol = tol,
      irls_max_iters = as.integer(1e6),
      max_iters = as.integer(1e8)
    )),
    error = function(e) NULL
  )
  runtime <- proc.time()[["elapsed"]] - t0
  if (is.null(state) || nrow(state$state$betas) == 0) {
    cat(sprintf("  tol=%.0e  FAILED\n", tol))
    next
  }
  beta <- as.numeric(state$state$betas[1, ])
  beta0 <- as.numeric(state$state$intercepts[1])
  primal <- logistic_primal(beta0, beta)
  cat(sprintf("  tol=%.0e  a0=%.4f  primal=%.10f  time=%.3fs\n",
              tol, beta0, primal, runtime))
  rows[[length(rows) + 1]] <- data.frame(
    tol = tol, primal = primal, runtime = runtime
  )
}

out <- do.call(rbind, rows)
write.csv(out, file.path(outdir, "adelie.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(outdir, "adelie.csv")))
