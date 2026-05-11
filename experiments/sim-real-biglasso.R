## Fig B: biglasso Newton vs MM on the shared imbalanced logistic problem.
## Same design as the glmnet driver: sweep convergence tolerance `eps`,
## record (npasses, primal) at termination.
##
## biglasso defaults to alg.logistic = "Newton" (Bucket 2, local-IRLS).
## alg.logistic = "MM" uses w = 1/4 majorant (Bucket 3); prediction: stalls /
## takes far more passes than Newton at high imbalance.

suppressPackageStartupMessages({
  library(biglasso)
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

X_big <- as.big.matrix(X)

logistic_primal <- function(beta0, beta) {
  eta <- as.numeric(beta0 + X %*% beta)
  z <- -y_pm * eta
  loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  mean(loss) + lambda * sum(abs(beta))
}

eps_grid <- 10 ^ seq(-1, -13, by = -0.5)
algs <- c("Newton", "MM")

rows <- list()
for (alg in algs) {
  cat(sprintf("=== alg.logistic = %s ===\n", alg))
  for (eps in eps_grid) {
    t0 <- proc.time()[["elapsed"]]
    fit <- tryCatch(
      suppressWarnings(biglasso::biglasso(
        X_big, y01,
        family = "binomial",
        lambda = lambda,
        eps = eps,
        max.iter = 1e6,
        screen = "None",
        alg.logistic = alg,
        warn = FALSE
      )),
      error = function(e) NULL
    )
    runtime <- proc.time()[["elapsed"]] - t0
    if (is.null(fit)) {
      cat(sprintf("  eps=%.0e  FAILED\n", eps))
      next
    }
    beta0 <- as.numeric(fit$beta[1])
    beta <- as.numeric(fit$beta[-1])
    primal <- logistic_primal(beta0, beta)
    npasses <- if (!is.null(fit$iter)) sum(fit$iter) else NA_integer_
    cat(sprintf("  eps=%.0e  npasses=%6s  a0=%.4f  primal=%.10f  time=%.3fs\n",
                eps, format(npasses), beta0, primal, runtime))
    rows[[length(rows) + 1]] <- data.frame(
      alg = alg, eps = eps, npasses = npasses,
      primal = primal, runtime = runtime
    )
  }
}

out <- do.call(rbind, rows)
write.csv(out, file.path(outdir, "biglasso.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(outdir, "biglasso.csv")))
