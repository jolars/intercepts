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
## Converts the shipped `data/yeoh/Yeoh2002.rds` (retrieved via
## `experiments/fetch-data.sh`) into the plain CSV inputs the Julia driver
## consumes. Run once before `experiments/sim-real-multinomial-yeoh.jl`; output
## goes to `data/yeoh/`, which is gitignored. This script no longer downloads
## anything --- the RDS ships in the pinned data archive.

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

# Provenance of the shipped RDS, recorded in meta.json. The file originates
# from the IowaBiostat data-sets collection and is redistributed in the pinned
# data archive; it is no longer fetched at run time.
rds_source <- paste0(
  "Yeoh2002 (St. Jude), originally ",
  "https://github.com/IowaBiostat/data-sets (Yeoh2002/Yeoh2002.rds); ",
  "redistributed in the pinned data archive on Zenodo"
)
rds_path <- file.path(out_dir, "Yeoh2002.rds")

if (!file.exists(rds_path)) {
  stop(
    "Yeoh2002.rds not found at ", rds_path, ".\n",
    "Retrieve the real-data inputs first with `bash experiments/fetch-data.sh` ",
    "(see the README's \"Retrieve the real-data inputs\" section)."
  )
}
message("Using shipped ", rds_path)

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
    source = rds_source
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
