## Fig B: biglasso Newton vs MM. Same design as the glmnet driver: warm-start
## down a 50-point path from lambda_max to reg*lambda_max with the requested
## eps, then record (total npasses across the path, primal at the final
## lambda). Path mode sidesteps the same single-lambda convergence pathology
## that affects glmnet on w1a's extreme imbalance + sparse-binary features
## (jerr=-1 / null solution) and matches glmnet's work axis directly.
##
## biglasso defaults to alg.logistic = "Newton" (Bucket B, local-IRLS).
## alg.logistic = "MM" uses w = 1/4 majorant (Bucket A); prediction: stalls /
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

run_sweep <- function(probdir, problem) {
  X <- as.matrix(read.csv(file.path(probdir, "X.csv"), header = FALSE))
  ydf <- read.csv(file.path(probdir, "y.csv"))
  y01 <- ydf$y01
  y_pm <- ydf$y_pm
  meta <- fromJSON(file.path(probdir, "meta.json"))
  reg <- meta$reg

  X_big <- as.big.matrix(X)

  logistic_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    z <- -y_pm * eta
    loss <- ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
    mean(loss) + lambda * sum(abs(beta))
  }

  eps_grid <- 10 ^ seq(-1, -13, by = -0.5)
  algs <- c("Newton", "MM")

  rows <- list()
  for (alg in algs) {
    cat(sprintf("=== problem=%s alg.logistic = %s ===\n", problem, alg))
    for (eps in eps_grid) {
      t0 <- proc.time()[["elapsed"]]
      fit <- tryCatch(
        suppressWarnings(biglasso::biglasso(
          X_big, y01,
          family = "binomial",
          nlambda = 50,
          lambda.min = reg,
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
      i <- length(fit$lambda)
      lam <- fit$lambda[i]
      beta0 <- as.numeric(fit$beta[1, i])
      beta <- as.numeric(fit$beta[-1, i])
      primal <- logistic_primal(beta0, beta, lam)
      npasses <- if (!is.null(fit$iter)) sum(fit$iter) else NA_integer_
      cat(sprintf(
        "  eps=%.0e  npasses=%6s  a0=%.4f  primal=%.10f  finalLam=%.4g  time=%.3fs\n",
        eps, format(npasses), beta0, primal, lam, runtime
      ))
      rows[[length(rows) + 1]] <- data.frame(
        problem = problem, alg = alg, eps = eps, npasses = npasses,
        primal = primal, final_lambda = lam, runtime = runtime
      )
    }
  }

  do.call(rbind, rows)
}

synthetic_dir <- file.path(root, "results", "real-solvers")
w1a_dir       <- file.path(root, "results", "real-solvers", "w1a")
news20_dir    <- file.path(root, "results", "real-solvers", "news20-3pct")

out_synth <- run_sweep(synthetic_dir, "synthetic")
write.csv(out_synth, file.path(synthetic_dir, "biglasso.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(synthetic_dir, "biglasso.csv")))

out_w1a <- run_sweep(w1a_dir, "w1a")
write.csv(out_w1a, file.path(w1a_dir, "biglasso.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(w1a_dir, "biglasso.csv")))

out_news20 <- run_sweep(news20_dir, "news20-3pct")
write.csv(out_news20, file.path(news20_dir, "biglasso.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(news20_dir, "biglasso.csv")))
