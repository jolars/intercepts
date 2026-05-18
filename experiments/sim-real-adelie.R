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
## disable adelie's path-level early-exit so every requested lambda is run
## to convergence.
##
## adelie is run in path mode (warm-start from lambda_max down to
## reg*lambda_max in 50 geometric steps) to match the glmnet/biglasso path
## drivers and to sidestep the same cold-start single-lambda overshoot that
## affects unguarded-Newton intercept updates at extreme imbalance.

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

## adelie's `tol` is dimensionless relative to the deviance gap (effective
## tolerance ~ tol * (loss_null - loss_full) / hess_sum inside the pin solver),
## so tol=3e-4 already drives suboptimality to ~7e-5 on this problem. Tighter
## than ~3e-4 the outer IRLS criterion `|Σ (resid - resid_prev)·(eta - eta_prev)|
## ≤ irls_tol` runs out of progress for a long time before the iterate cap,
## so each tol below the wall takes many minutes (and may fail outright by
## returning a state with zero stored λs). The working range gives a clean
## three-order-of-magnitude span in suboptimality.
tol_grid <- 10 ^ seq(-1, -3.5, by = -0.25)

run_sweep <- function(probdir, problem) {
  X <- as.matrix(read.csv(file.path(probdir, "X.csv"), header = FALSE))
  ydf <- read.csv(file.path(probdir, "y.csv"))
  y01 <- ydf$y01
  y_pm <- ydf$y_pm
  meta <- fromJSON(file.path(probdir, "meta.json"))
  reg <- meta$reg
  lambda_max <- meta$lambda_max
  lambda_path <- lambda_max * exp(seq(0, log(reg), length.out = 50))

  logistic_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    z <- -y_pm * eta
    loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
    mean(loss) + lambda * sum(abs(beta))
  }

  rows <- list()
  cat(sprintf("=== adelie problem=%s ===\n", problem))
  for (tol in tol_grid) {
    t0 <- proc.time()[["elapsed"]]
    state <- tryCatch(
      suppressWarnings(adelie::grpnet(
        X = X,
        glm = adelie::glm.binomial(y = y01),
        lambda = lambda_path,
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
    i <- nrow(state$state$betas)
    lam <- state$state$lmdas[i]
    beta <- as.numeric(state$state$betas[i, ])
    beta0 <- as.numeric(state$state$intercepts[i])
    primal <- logistic_primal(beta0, beta, lam)
    cat(sprintf(
      "  tol=%.0e  a0=%.4f  primal=%.10f  finalLam=%.4g  time=%.3fs\n",
      tol, beta0, primal, lam, runtime
    ))
    rows[[length(rows) + 1]] <- data.frame(
      problem = problem, tol = tol, primal = primal,
      final_lambda = lam, runtime = runtime
    )
  }

  do.call(rbind, rows)
}

synthetic_dir <- file.path(root, "results", "real-solvers")
w1a_dir       <- file.path(root, "results", "real-solvers", "w1a")
news20_dir    <- file.path(root, "results", "real-solvers", "news20-3pct")

out_synth <- run_sweep(synthetic_dir, "synthetic")
write.csv(out_synth, file.path(synthetic_dir, "adelie.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(synthetic_dir, "adelie.csv")))

out_w1a <- run_sweep(w1a_dir, "w1a")
write.csv(out_w1a, file.path(w1a_dir, "adelie.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(w1a_dir, "adelie.csv")))

out_news20 <- run_sweep(news20_dir, "news20-3pct")
write.csv(out_news20, file.path(news20_dir, "adelie.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(news20_dir, "adelie.csv")))
