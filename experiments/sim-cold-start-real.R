## Single-lambda cold-start diagnostic on real imbalanced logistic problems
## (w1a and news20-3pct). Tests whether the
## unguarded-Newton intercept overshoot that affects biglasso in path mode at
## the cold-start step also bites at single-lambda with a loose eps, and
## whether it reproduces across both datasets (matched on ybar = 0.03, very
## different X structure: 300-feature web indicators vs 4675-feature bag of
## words). The argument the paper would make from this: the intercept-specific
## fix (MM majorant w = 1/4) is X-free and converges in O(1) iterations on
## the same problem, so its absence in production solvers is a missed
## opportunity, not a fundamental limitation.
##
## Setup:
##   - Single target lambda = reg * lambda_max from meta.json (no path
##     warm-start to absorb the cold-start transient).
##   - eps = 1e-1 (loose -- a single converged step on the working quadratic
##     should already meet this).
##   - Sweep max.iter from 10 to 1e5 in decade steps to see the trajectory.
##   - Compare biglasso alg.logistic = "Newton" (unguarded) vs "MM" (w = 1/4).
##   - Same sweep on adelie's outer IRLS (irls_max_iters).

suppressPackageStartupMessages({
  library(biglasso)
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

run_diag <- function(probdir, problem) {
  X <- as.matrix(read.csv(file.path(probdir, "X.csv"), header = FALSE))
  ydf <- read.csv(file.path(probdir, "y.csv"))
  y01 <- ydf$y01
  y_pm <- ydf$y_pm
  meta <- fromJSON(file.path(probdir, "meta.json"))
  lambda_target <- meta$lambda

  X_big <- as.big.matrix(X)

  logistic_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    z <- -y_pm * eta
    loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
    mean(loss) + lambda * sum(abs(beta))
  }

  iter_grid <- c(10, 100, 1000, 10000, 100000)
  rows <- list()

  ## biglasso: Newton (unguarded local-IRLS) vs MM (w = 1/4 majorant)
  for (alg in c("Newton", "MM")) {
    cat(sprintf("=== %s biglasso alg=%s (single-lambda cold start) ===\n",
                problem, alg))
    for (mi in iter_grid) {
      t0 <- proc.time()[["elapsed"]]
      fit <- tryCatch(
        suppressWarnings(biglasso::biglasso(
          X_big, y01,
          family = "binomial",
          lambda = lambda_target,
          eps = 1e-1,
          max.iter = mi,
          screen = "None",
          alg.logistic = alg,
          warn = FALSE
        )),
        error = function(e) conditionMessage(e)
      )
      runtime <- proc.time()[["elapsed"]] - t0
      if (is.character(fit)) {
        cat(sprintf("  max.iter=%6d  FAILED: %s  time=%.2fs\n",
                    mi, fit, runtime))
        rows[[length(rows) + 1]] <- data.frame(
          problem = problem, solver = "biglasso", alg = alg,
          max_iter = mi, beta0 = NA_real_, primal = NA_real_,
          l1_beta = NA_real_, npasses = NA_integer_,
          runtime = runtime, status = "FAIL"
        )
        next
      }
      beta0 <- as.numeric(fit$beta[1, 1])
      beta <- as.numeric(fit$beta[-1, 1])
      primal <- logistic_primal(beta0, beta, lambda_target)
      npasses <- if (!is.null(fit$iter)) sum(fit$iter) else NA_integer_
      cat(sprintf(
        "  max.iter=%6d  beta0=%10.4f  primal=%.6e  l1=%8.3f  npasses=%6s  time=%.2fs\n",
        mi, beta0, primal, sum(abs(beta)), format(npasses), runtime
      ))
      rows[[length(rows) + 1]] <- data.frame(
        problem = problem, solver = "biglasso", alg = alg,
        max_iter = mi, beta0 = beta0, primal = primal,
        l1_beta = sum(abs(beta)), npasses = npasses,
        runtime = runtime, status = "OK"
      )
    }
  }

  ## adelie: outer IRLS with default unguarded Newton step
  cat(sprintf("=== %s adelie (single-lambda cold start) ===\n", problem))
  for (mi in iter_grid) {
    t0 <- proc.time()[["elapsed"]]
    state <- tryCatch(
      suppressWarnings(adelie::grpnet(
        X = X,
        glm = adelie::glm.binomial(y = y01),
        lambda = lambda_target,
        intercept = TRUE,
        standardize = FALSE,
        early_exit = FALSE,
        adev_tol = 1.0,
        tol = 1e-1,
        irls_tol = 1e-1,
        irls_max_iters = as.integer(mi),
        max_iters = as.integer(max(mi, 1e6))
      )),
      error = function(e) conditionMessage(e)
    )
    runtime <- proc.time()[["elapsed"]] - t0
    if (is.character(state)) {
      cat(sprintf("  irls_max_iters=%6d  FAILED: %s  time=%.2fs\n",
                  mi, state, runtime))
      rows[[length(rows) + 1]] <- data.frame(
        problem = problem, solver = "adelie", alg = "default",
        max_iter = mi, beta0 = NA_real_, primal = NA_real_,
        l1_beta = NA_real_, npasses = NA_integer_,
        runtime = runtime, status = "FAIL"
      )
      next
    }
    if (nrow(state$state$betas) == 0) {
      cat(sprintf("  irls_max_iters=%6d  EMPTY (no lambda solved)  time=%.2fs\n",
                  mi, runtime))
      rows[[length(rows) + 1]] <- data.frame(
        problem = problem, solver = "adelie", alg = "default",
        max_iter = mi, beta0 = NA_real_, primal = NA_real_,
        l1_beta = NA_real_, npasses = NA_integer_,
        runtime = runtime, status = "EMPTY"
      )
      next
    }
    beta <- as.numeric(state$state$betas[1, ])
    beta0 <- as.numeric(state$state$intercepts[1])
    primal <- logistic_primal(beta0, beta, lambda_target)
    cat(sprintf(
      "  irls_max_iters=%6d  beta0=%10.4f  primal=%.6e  l1=%8.3f  time=%.2fs\n",
      mi, beta0, primal, sum(abs(beta)), runtime
    ))
    rows[[length(rows) + 1]] <- data.frame(
      problem = problem, solver = "adelie", alg = "default",
      max_iter = mi, beta0 = beta0, primal = primal,
      l1_beta = sum(abs(beta)), npasses = NA_integer_,
      runtime = runtime, status = "OK"
    )
  }

  do.call(rbind, rows)
}

w1a_dir    <- file.path(root, "results", "real-solvers", "w1a")
news20_dir <- file.path(root, "results", "real-solvers", "news20-3pct")

out_w1a    <- run_diag(w1a_dir, "w1a")
out_news20 <- run_diag(news20_dir, "news20-3pct")
out <- rbind(out_w1a, out_news20)

outfile <- file.path(root, "results", "real-solvers", "cold-start-diag.csv")
write.csv(out, outfile, row.names = FALSE)
cat(sprintf("Wrote %s\n", outfile))
