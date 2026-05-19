## adelie on the congress109 Poisson problem.
##
## adelie (Yang & Hastie 2024) is a direct-Newton / IRLS solver: each outer
## pass linearises the non-Gaussian loss into a weighted Gaussian surrogate,
## the inner CD pin solver minimises the surrogate, and the intercept is
## implicit in the centered-features parameterisation (so it sits at the
## optimum of the surrogate by construction). For Poisson this is the same
## bucket as glmnet's local-IRLS mode.
##
## Run in path mode (warm-start lambda_max down to reg*lambda_max in 50
## geometric steps), `tol = irls_tol = tol_grid[k]`, all other knobs as in
## sim-real-adelie.R. Output: (tol, primal, runtime) per tolerance.

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

tol_grid <- 10 ^ seq(-1, -3.5, by = -0.25)

run_sweep <- function(probdir, problem) {
  X <- as.matrix(read.csv(file.path(probdir, "X.csv"), header = FALSE))
  ydf <- read.csv(file.path(probdir, "y.csv"))
  y <- ydf$y
  meta <- fromJSON(file.path(probdir, "meta.json"))
  reg <- meta$reg
  lambda_max <- meta$lambda_max
  lambda_path <- lambda_max * exp(seq(0, log(reg), length.out = 50))

  poisson_primal <- function(beta0, beta, lambda) {
    eta <- as.numeric(beta0 + X %*% beta)
    mean(exp(eta) - y * eta) + lambda * sum(abs(beta))
  }

  rows <- list()
  cat(sprintf("=== adelie Poisson problem=%s ===\n", problem))
  for (tol in tol_grid) {
    t0 <- proc.time()[["elapsed"]]
    state <- tryCatch(
      suppressWarnings(adelie::grpnet(
        X = X,
        glm = adelie::glm.poisson(y = y),
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
    primal <- poisson_primal(beta0, beta, lam)
    cat(sprintf(
      "  tol=%.0e  a0=%.4f  primal=%.10f  finalLam=%.4g  time=%.3fs\n",
      tol, beta0, primal, lam, runtime
    ))
    rows[[length(rows) + 1]] <- data.frame(
      problem = problem,
      tol = tol,
      primal = primal,
      final_lambda = lam,
      runtime = runtime
    )
  }

  do.call(rbind, rows)
}

congress_dir <- file.path(root, "results", "real-solvers-poisson", "congress109")
out <- run_sweep(congress_dir, "congress109")
write.csv(out, file.path(congress_dir, "adelie.csv"), row.names = FALSE)
cat(sprintf("Wrote %s\n", file.path(congress_dir, "adelie.csv")))
