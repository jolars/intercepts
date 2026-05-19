## glmnet on the congress109 Poisson problem.
##
## Poisson loss has no global upper bound on f''(eta) = exp(eta), so unlike
## the binomial family there is no MM/upper-bound surrogate available --
## glmnet has only one fitting mode for `family = "poisson"` (local-IRLS).
## That is itself the point of the panel: the within-solver Newton-vs-MM
## comparison that the logistic figures show is absent here, because no
## production solver has an MM analog for Poisson.
##
## We sweep convergence tolerance `thresh` from loose to tight on path mode
## (warm-start from lambda_max down to reg*lambda_max via a 50-point
## geometric sequence), with standardize=FALSE since X on disk is already
## standardised. Output mirrors sim-real-glmnet.R: a (thresh, npasses,
## primal, runtime) row per tolerance.

suppressPackageStartupMessages({
  library(glmnet)
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

run_sweep <- function(probdir, problem) {
  X <- as.matrix(read.csv(file.path(probdir, "X.csv"), header = FALSE))
  ydf <- read.csv(file.path(probdir, "y.csv"))
  y <- ydf$y
  meta <- fromJSON(file.path(probdir, "meta.json"))
  reg <- meta$reg

  poisson_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    mean(exp(eta) - y * eta) + lambda * sum(abs(beta))
  }

  thresh_grid <- 10 ^ seq(-1, -13, by = -0.5)

  rows <- list()
  cat(sprintf("=== glmnet Poisson problem=%s ===\n", problem))
  for (thresh in thresh_grid) {
    t0 <- proc.time()[["elapsed"]]
    fit <- suppressWarnings(glmnet::glmnet(
      X, y,
      family = "poisson",
      nlambda = 50,
      lambda.min.ratio = reg,
      intercept = TRUE,
      standardize = FALSE,
      thresh = thresh,
      maxit = 1e6
    ))
    runtime <- proc.time()[["elapsed"]] - t0
    i <- length(fit$lambda)
    a0 <- fit$a0[i]
    b <- as.numeric(fit$beta[, i])
    lam <- fit$lambda[i]
    primal <- poisson_primal(a0, b, lam)
    cat(sprintf(
      "  thresh=%.0e  npasses=%5d  primal=%.10f  finalLam=%.4g  t=%.3fs  jerr=%d\n",
      thresh, fit$npasses, primal, lam, runtime, fit$jerr
    ))
    rows[[length(rows) + 1]] <- data.frame(
      problem = problem,
      thresh = thresh,
      npasses = fit$npasses,
      primal = primal,
      final_lambda = lam,
      runtime = runtime,
      jerr = fit$jerr
    )
  }

  do.call(rbind, rows)
}

congress_dir <- file.path(root, "results", "real-solvers-poisson", "congress109")
out <- run_sweep(congress_dir, "congress109")
write.csv(out, file.path(congress_dir, "glmnet.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(congress_dir, "glmnet.csv")))
