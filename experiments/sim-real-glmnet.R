## Fig A: glmnet Newton vs modified.Newton on the shared imbalanced logistic
## problem. Sweeps the convergence tolerance `thresh` from loose to tight (with
## maxit large enough not to bind), records (npasses, primal) at termination.
##
## glmnet's `maxit` interface returns the null solution when binding (jerr=-1),
## so it is not a useful work axis. Instead we treat thresh as the knob: each
## (thresh, type.logistic) pair is one operating point on a work-quality curve.
##
## The paper predicts the curves separate: Newton reaches a given subopt with
## far fewer passes than modified.Newton at high imbalance.

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

thresh_grid <- 10 ^ seq(-1, -13, by = -0.5)
types <- c("Newton", "modified.Newton")

rows <- list()
for (type in types) {
  cat(sprintf("=== type.logistic = %s ===\n", type))
  for (thresh in thresh_grid) {
    t0 <- proc.time()[["elapsed"]]
    fit <- suppressWarnings(glmnet::glmnet(
      X, y01,
      family = "binomial",
      lambda = lambda,
      intercept = TRUE,
      standardize = FALSE,
      thresh = thresh,
      maxit = 1e6,
      type.logistic = type
    ))
    runtime <- proc.time()[["elapsed"]] - t0
    beta0 <- as.numeric(fit$a0)
    beta <- as.numeric(fit$beta)
    primal <- logistic_primal(beta0, beta)
    cat(sprintf("  thresh=%.0e  npasses=%5d  primal=%.10f  time=%.3fs  jerr=%d\n",
                thresh, fit$npasses, primal, runtime, fit$jerr))
    rows[[length(rows) + 1]] <- data.frame(
      type = type, thresh = thresh, npasses = fit$npasses,
      primal = primal, runtime = runtime, jerr = fit$jerr
    )
  }
}

out <- do.call(rbind, rows)
write.csv(out, file.path(outdir, "glmnet.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(outdir, "glmnet.csv")))
