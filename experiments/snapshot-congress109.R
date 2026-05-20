## Author-only: snapshot the congress109 phrase-count matrix from R's textir
## package into a pinned, language-neutral form (MatrixMarket + a phrase list)
## under `data/congress109/`. The snapshot is the *input* shipped in the data
## archive; `sim-real-poisson-problem.R` reads it instead of loading textir at
## run time, so the Poisson problem reproduces without depending on the textir
## data object being present.
##
## Run via `experiments/build-data-archive.jl`; not needed to reproduce results
## once the archive has been retrieved.

suppressPackageStartupMessages({
  library(textir)
  library(Matrix)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  flag <- "--file="
  match <- grep(flag, args)
  if (length(match) > 0) {
    normalizePath(dirname(sub(flag, "", args[match])))
  } else {
    normalizePath(".")
  }
}
root <- normalizePath(file.path(get_script_dir(), ".."))

data(congress109, package = "textir")
counts <- as(congress109Counts, "CsparseMatrix")

out_dir <- file.path(root, "data", "congress109")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# MatrixMarket preserves the exact (i, j, value) triplets and the column order,
# so the deterministic target-phrase pick (which.max of column sums) in
# sim-real-poisson-problem.R lands on the same phrase as it does on the textir
# object. The phrase vocabulary is shipped separately because writeMM drops
# dimnames.
Matrix::writeMM(counts, file.path(out_dir, "counts.mtx"))
writeLines(colnames(counts), file.path(out_dir, "phrases.txt"))

cat(sprintf(
  "Wrote congress109 snapshot to %s (%d speakers x %d phrases)\n",
  out_dir, nrow(counts), ncol(counts)
))
