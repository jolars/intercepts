## glmnet Newton vs modified.Newton on the shared synthetic imbalanced
## logistic problem and on w1a (a severely imbalanced LIBSVM problem). Both
## fits use glmnet's recommended *path mode* (warm-start from lambda_max
## down to reg*lambda_max via an nlambda-point geometric sequence), with
## standardize=TRUE so that glmnet applies its own internal feature
## scaling. We use path mode because glmnet's single-lambda fit does not
## converge on w1a's extreme imbalance + sparse-binary features at any
## reasonable lambda (jerr=-1 / null solution); the path's warm-starts
## sidestep that pathology. The comparison axis is therefore total CD
## passes to reach the final (smallest) lambda = reg*lambda_max.
##
## We sweep convergence tolerance `thresh` from loose to tight and record
## (npasses, primal) at termination of the full path. Each (thresh, type)
## pair is one operating point on a work-quality curve.
##
## The paper predicts the curves separate: Newton reaches a given subopt
## with far fewer passes than modified.Newton at high imbalance. The w1a
## panel checks the gap is a property of the intercept update, not of the
## synthetic construction.

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
  y01 <- ydf$y01
  y_pm <- ydf$y_pm
  meta <- fromJSON(file.path(probdir, "meta.json"))
  reg <- meta$reg

  logistic_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    z <- -y_pm * eta
    loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
    mean(loss) + lambda * sum(abs(beta))
  }

  thresh_grid <- 10 ^ seq(-1, -13, by = -0.5)
  types <- c("Newton", "modified.Newton")

  rows <- list()
  for (type in types) {
    cat(sprintf("=== problem=%s type.logistic = %s ===\n", problem, type))
    for (thresh in thresh_grid) {
      t0 <- proc.time()[["elapsed"]]
      fit <- suppressWarnings(glmnet::glmnet(
        X, y01,
        family = "binomial",
        nlambda = 50,
        lambda.min.ratio = reg,
        intercept = TRUE,
        standardize = TRUE,
        thresh = thresh,
        maxit = 1e6,
        type.logistic = type
      ))
      runtime <- proc.time()[["elapsed"]] - t0
      i <- length(fit$lambda)
      a0 <- fit$a0[i]
      b <- as.numeric(fit$beta[, i])
      lam <- fit$lambda[i]
      primal <- logistic_primal(a0, b, lam)
      cat(sprintf("  thresh=%.0e  npasses=%5d  primal=%.10f  finalLam=%.4g  t=%.3fs  jerr=%d\n",
                  thresh, fit$npasses, primal, lam, runtime, fit$jerr))
      rows[[length(rows) + 1]] <- data.frame(
        problem = problem, type = type, thresh = thresh,
        npasses = fit$npasses, primal = primal,
        final_lambda = lam, runtime = runtime, jerr = fit$jerr
      )
    }
  }

  do.call(rbind, rows)
}

synthetic_dir <- file.path(root, "results", "real-solvers")
w1a_dir       <- file.path(root, "results", "real-solvers", "w1a")

out_synth <- run_sweep(synthetic_dir, "synthetic")
write.csv(out_synth, file.path(synthetic_dir, "glmnet.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(synthetic_dir, "glmnet.csv")))

out_w1a <- run_sweep(w1a_dir, "w1a")
write.csv(out_w1a, file.path(w1a_dir, "glmnet.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(w1a_dir, "glmnet.csv")))
