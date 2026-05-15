## Fetch the Yeoh2002 pediatric ALL gene-expression dataset from the Iowa
## Biostatistics dataset collection and stage it as plain CSV for the Julia
## experiment driver to consume.
##
## Yeoh2002 carries six subtypes of B-precursor ALL on a 248-sample microarray
## panel with 12 625 gene expression features. The class marginal ranges from
## 6 % (BCR) to 32 % (TEL), giving a rare-class regime in genuine
## high-dimensional (p > n) gene-expression space -- the real-data analogue
## of @fig-multinomial-imbalance.
##
## Run once before `experiments/sim-real-multinomial-yeoh.jl`; output goes to
## `data/yeoh/`, which is gitignored.

suppressPackageStartupMessages({
  library(jsonlite)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  flag <- "--file="
  match <- grep(flag, args)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(flag, "", args[match]))))
  }
  return(getwd())
}

project_root <- normalizePath(file.path(get_script_dir(), ".."))
out_dir <- file.path(project_root, "data", "yeoh")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rds_url <- "https://github.com/IowaBiostat/data-sets/raw/main/Yeoh2002/Yeoh2002.rds"
rds_path <- file.path(out_dir, "Yeoh2002.rds")

if (!file.exists(rds_path)) {
  message("Downloading Yeoh2002.rds (~33 MB) ...")
  download.file(rds_url, rds_path, mode = "wb", quiet = FALSE)
} else {
  message("Using cached ", rds_path)
}

data <- readRDS(rds_path)
stopifnot(all(c("X", "y") %in% names(data)))
X <- data$X
y <- data$y

stopifnot(is.matrix(X))
stopifnot(is.factor(y))

class_labels <- levels(y)
y_int <- as.integer(y)  # 1..K following the factor's level order

x_path <- file.path(out_dir, "X.csv.gz")
y_path <- file.path(out_dir, "y.csv")
meta_path <- file.path(out_dir, "meta.json")

con <- gzfile(x_path, "w")
write.table(
  X,
  file = con,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
close(con)

write.table(
  y_int,
  file = y_path,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

class_counts <- as.integer(table(y))
names(class_counts) <- class_labels

write_json(
  list(
    n = nrow(X),
    p = ncol(X),
    K = length(class_labels),
    class_labels = class_labels,
    class_counts = as.list(class_counts),
    source = rds_url
  ),
  meta_path,
  pretty = TRUE,
  auto_unbox = TRUE
)

message("Wrote ", x_path)
message("Wrote ", y_path)
message("Wrote ", meta_path)
message("X dim: ", nrow(X), " x ", ncol(X))
message("Class counts: ", paste(class_labels, class_counts, sep = "=", collapse = ", "))
