suppressPackageStartupMessages({
  library(glmnet)
  library(biglasso)
  library(jsonlite)
})

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg)), ".."))
outdir <- file.path(root, "results", "production-imbalance")
problems <- read.csv(file.path(outdir, "problems.csv"))
arguments <- commandArgs(trailingOnly = TRUE)
resume <- "--resume" %in% arguments
requested_cells <- setdiff(arguments, "--resume")
if (length(requested_cells)) {
  problems <- problems[problems$cell %in% requested_cells, ]
}
stopifnot(nrow(problems) > 0)
if (!length(requested_cells)) {
  stopifnot(nrow(problems) == 15L)
}

target <- 1e-6
repetitions <- 7L
tolerance_grid <- 10^seq(-3, -13, by = -0.5)
configs <- data.frame(
  package = c("glmnet", "glmnet", "biglasso", "biglasso"),
  mode = c("Newton", "modified.Newton", "Newton", "MM")
)
dir.create(file.path(outdir, "cells"), showWarnings = FALSE)

primal_value <- function(X, y, beta0, beta, lambda) {
  eta <- drop(beta0 + X %*% beta)
  z <- (1 - 2 * y) * eta
  mean(pmax(z, 0) + log1p(exp(-abs(z)))) + lambda * sum(abs(beta))
}

fit_path <- function(config, tolerance, X, X_big, y, lambdas) {
  if (config$package == "glmnet") {
    fit <- glmnet::glmnet(
      X,
      y,
      family = "binomial",
      lambda = lambdas,
      standardize = FALSE,
      intercept = TRUE,
      type.logistic = config$mode,
      control = list(thresh = tolerance, maxit = 1e6, fdev = 0, devmax = 1)
    )
  } else {
    fit <- biglasso::biglasso(
      X_big,
      y,
      family = "binomial",
      lambda = lambdas,
      eps = tolerance,
      max.iter = 10000,
      alg.logistic = config$mode,
      screen = "SSR",
      ncores = 1,
      warn = TRUE
    )
  }
  fit
}

assess <- function(fit, package, X, y, problem, lambdas) {
  incomplete <- list(
    primal = NA_real_,
    relative_bound = Inf,
    npasses = NA_real_
  )
  if (length(fit$lambda) != length(lambdas)) {
    return(incomplete)
  }
  stopifnot(max(abs(fit$lambda - lambdas)) < 1e-12)
  i <- length(lambdas)
  if (package == "glmnet") {
    if (fit$jerr != 0) {
      return(incomplete)
    }
    beta0 <- fit$a0[i]
    beta <- as.numeric(fit$beta[, i])
    passes <- fit$npasses
  } else {
    # Internal standardization must preserve the shared penalty convention.
    stopifnot(max(abs(fit$scale - 1)) < 1e-12, max(abs(fit$center)) < 1e-12)
    if (!all(is.finite(fit$iter))) {
      return(incomplete)
    }
    beta0 <- as.numeric(fit$beta[1, i])
    beta <- as.numeric(fit$beta[-1, i])
    passes <- sum(fit$iter)
  }
  if (!all(is.finite(c(beta0, beta)))) {
    return(incomplete)
  }
  primal <- primal_value(X, y, beta0, beta, problem$lambda)
  gap <- primal - problem$dual_bound
  stopifnot(is.finite(primal), gap >= -1e-12)
  list(
    primal = primal,
    relative_bound = max(gap, 0) / problem$reference_primal,
    npasses = passes
  )
}

for (cell in problems$cell) {
  problem <- problems[problems$cell == cell, ]
  design_path <- file.path(outdir, "inputs", problem$design_file)
  response_path <- file.path(outdir, "inputs", problem$response_file)
  timing_path <- file.path(outdir, "cells", paste0(cell, "-timings.csv"))
  calibration_path <- file.path(
    outdir,
    "cells",
    paste0(cell, "-calibration.csv")
  )
  if (resume && file.exists(timing_path) && file.exists(calibration_path)) {
    saved <- read.csv(
      timing_path,
      colClasses = c(package_version = "character")
    )
    saved_calibration <- read.csv(calibration_path)
    reached <- unique(saved_calibration[
      saved_calibration$relative_bound <= target,
      c("package", "mode")
    ])
    current_versions <- vapply(
      saved$package,
      function(p) as.character(packageVersion(p)),
      ""
    )
    inputs_changed <- max(
      file.info(c(
        design_path,
        response_path,
        file.path(outdir, "problems.csv")
      ))$mtime
    ) >
      min(file.info(c(timing_path, calibration_path))$mtime)
    saved_bound <- (saved$primal - problem$dual_bound) /
      problem$reference_primal
    if (
      !inputs_changed &&
        nrow(saved) > 0 &&
        nrow(saved) == repetitions * nrow(reached) &&
        all(saved$target == target) &&
        all(saved$package_version == current_versions) &&
        all(saved_bound >= 0 & saved_bound <= target) &&
        max(abs(saved_bound - saved$relative_bound)) < 1e-12
    ) {
      cat(cell, "reusing completed timing cell\n")
      flush.console()
      next
    }
  }
  X <- as.matrix(read.csv(design_path, header = FALSE))
  y <- read.csv(response_path)$y
  X_big <- as.big.matrix(X)
  lambdas <- exp(seq(
    log(problem$lambda_max),
    log(problem$lambda),
    length.out = 50
  ))
  tolerances <- numeric(nrow(configs))
  calibration <- list()

  # Select by certified accuracy before timing, so timing noise cannot choose
  # the tolerance or favor the fastest observed replicate.
  for (j in seq_len(nrow(configs))) {
    config <- configs[j, ]
    for (tolerance in tolerance_grid) {
      cat(
        cell,
        config$package,
        config$mode,
        "calibrating tolerance",
        tolerance,
        "\n"
      )
      flush.console()
      fit <- fit_path(config, tolerance, X, X_big, y, lambdas)
      accuracy <- assess(fit, config$package, X, y, problem, lambdas)
      calibration[[length(calibration) + 1L]] <- data.frame(
        cell = cell,
        package = config$package,
        mode = config$mode,
        tolerance = tolerance,
        target = target,
        accuracy
      )
      write.csv(
        do.call(rbind, calibration),
        calibration_path,
        row.names = FALSE
      )
      cat("  relative bound:", accuracy$relative_bound, "\n")
      flush.console()
      if (accuracy$relative_bound <= target) {
        tolerances[j] <- tolerance
        break
      }
    }
    if (tolerances[j] == 0) {
      cat(
        cell,
        config$package,
        config$mode,
        "did not reach the accuracy target\n"
      )
      flush.console()
      next
    }
    # Warm each selected fit once before collecting the timed repetitions.
    warm <- fit_path(config, tolerances[j], X, X_big, y, lambdas)
    stopifnot(
      assess(warm, config$package, X, y, problem, lambdas)$relative_bound <=
        target
    )
  }

  rows <- list()
  set.seed(1000L + 100L * problem$seed + round(100 * problem$prevalence))
  for (repetition in seq_len(repetitions)) {
    # Interleave modes to distribute drift in machine load across the pair.
    eligible <- which(tolerances > 0)
    order <- eligible[sample.int(length(eligible))]
    for (position in seq_along(order)) {
      j <- order[position]
      config <- configs[j, ]
      gc()
      started <- proc.time()[["elapsed"]]
      fit <- fit_path(config, tolerances[j], X, X_big, y, lambdas)
      runtime <- proc.time()[["elapsed"]] - started
      accuracy <- assess(fit, config$package, X, y, problem, lambdas)
      stopifnot(runtime > 0, accuracy$relative_bound <= target)
      rows[[length(rows) + 1L]] <- data.frame(
        cell = cell,
        seed = problem$seed,
        prevalence = problem$prevalence,
        package = config$package,
        mode = config$mode,
        repetition = repetition,
        execution_order = position,
        tolerance = tolerances[j],
        target = target,
        runtime = runtime,
        accuracy,
        package_version = as.character(packageVersion(config$package))
      )
    }
    write.csv(do.call(rbind, rows), timing_path, row.names = FALSE)
    cat(cell, "finished repetition", repetition, "of", repetitions, "\n")
    flush.console()
  }
  cell_rows <- do.call(rbind, rows)
  write.csv(
    cell_rows,
    file.path(outdir, "cells", paste0(cell, "-timings.csv")),
    row.names = FALSE
  )
  cat(
    cell,
    "finished; median times:",
    paste(
      round(
        tapply(
          cell_rows$runtime,
          paste(cell_rows$package, cell_rows$mode),
          median
        ),
        4
      ),
      collapse = ", "
    ),
    "\n"
  )
  flush.console()
}

timing_files <- file.path(
  outdir,
  "cells",
  paste0(problems$cell, "-timings.csv")
)
calibration_files <- file.path(
  outdir,
  "cells",
  paste0(problems$cell, "-calibration.csv")
)
timings <- do.call(
  rbind,
  lapply(timing_files, read.csv, colClasses = c(package_version = "character"))
)
calibration <- do.call(rbind, lapply(calibration_files, read.csv))
write.csv(timings, file.path(outdir, "timings.csv"), row.names = FALSE)
write.csv(calibration, file.path(outdir, "calibration.csv"), row.names = FALSE)

configurations <- do.call(
  rbind,
  lapply(
    split(
      calibration,
      interaction(
        calibration$cell,
        calibration$package,
        calibration$mode,
        drop = TRUE
      )
    ),
    function(rows) {
      successful <- rows[rows$relative_bound <= target, ]
      data.frame(
        cell = rows$cell[1],
        package = rows$package[1],
        mode = rows$mode[1],
        reached_target = nrow(successful) > 0,
        selected_tolerance = if (nrow(successful)) {
          successful$tolerance[1]
        } else {
          NA_real_
        },
        best_relative_bound = min(rows$relative_bound)
      )
    }
  )
)
write.csv(
  configurations,
  file.path(outdir, "configurations.csv"),
  row.names = FALSE
)

summary <- aggregate(
  runtime ~ cell + seed + prevalence + package + mode,
  data = timings,
  FUN = median
)
names(summary)[names(summary) == "runtime"] <- "median_runtime"
local <- summary[summary$mode == "Newton", ]
majorant <- summary[summary$mode != "Newton", ]
ratios <- merge(
  majorant,
  local,
  by = c("cell", "seed", "prevalence", "package"),
  suffixes = c("_majorant", "_newton"),
  all = TRUE
)
grid <- merge(
  problems[c("cell", "seed", "prevalence")],
  data.frame(package = unique(configs$package))
)
ratios <- merge(
  grid,
  ratios,
  by = c("cell", "seed", "prevalence", "package"),
  all.x = TRUE
)
ratios$runtime_ratio <- ratios$median_runtime_majorant /
  ratios$median_runtime_newton
ratios$comparison_available <- is.finite(ratios$runtime_ratio)
ratios <- merge(
  ratios,
  problems[c(
    "cell",
    "observed_prevalence",
    "curvature_fraction",
    "kappa",
    "max_rho_squared",
    "n_active"
  )],
  by = "cell"
)
write.csv(ratios, file.path(outdir, "ratios.csv"), row.names = FALSE)
write_json(
  list(
    target_relative_bound = target,
    repetitions = repetitions,
    tolerance_grid = tolerance_grid,
    path_length = 50,
    iteration_limits = list(glmnet = 1000000, biglasso = 10000),
    timing = "Elapsed full-path fit, excluding tolerance calibration, explicit warm-up, gc, input conversion, and certificate evaluation",
    glmnet = as.character(packageVersion("glmnet")),
    biglasso = as.character(packageVersion("biglasso")),
    R = R.version.string,
    thread_environment = as.list(Sys.getenv(c(
      "OPENBLAS_NUM_THREADS",
      "OMP_NUM_THREADS",
      "MKL_NUM_THREADS"
    ))),
    hardware = list(
      cpu = sub(
        ".*: *",
        "",
        grep("^model name", readLines("/proc/cpuinfo"), value = TRUE)[1]
      ),
      memory = grep("^MemTotal:", readLines("/proc/meminfo"), value = TRUE)
    ),
    session = capture.output(sessionInfo())
  ),
  file.path(outdir, "benchmark.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = NA
)
