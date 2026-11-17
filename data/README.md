# Real-data inputs

This directory holds the real datasets behind the paper's real-data experiments.
The raw files are gitignored. Retrieve the inputs with:

```bash
bash experiments/fetch-data.sh
```

The script retrieves the inputs into a temporary directory and verifies them
against the committed `data/MANIFEST.sha256` before installing them here,
preserving the repository's manifest and documentation. It then runs
`experiments/fetch-yeoh.R` to derive the Yeoh CSV inputs from the shipped RDS.
Only `MANIFEST.sha256`, `libsvm/manifest.json`, and this README are tracked in
git.

The cached results under `results/` already render the paper without any of
these inputs. You only need them to re-run the real-data experiments.

## Layout

  | Path                                                | Provenance                                                                                                                                                                                                                                                                                                                         |
  | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `yeoh/Yeoh2002.rds`                                 | Yeoh2002 pediatric ALL gene-expression panel (St. Jude); originally distributed via the IowaBiostat data-sets collection. `fetch-yeoh.R` converts it to `X.csv.gz`, `y.csv`, `meta.json`.                                                                                                                                          |
  | `libsvm/<name>`                                     | LIBSVM-format files for w1a, w8a, news20.binary, shuttle.scale[.t], a4a, leukemia, breast-cancer, and gisette_scale, from the [LIBSVM datasets collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). Dimensions and source URLs are recorded in `libsvm/manifest.json`; input checksums are in `MANIFEST.sha256`. |
  | `congress109/counts.mtx`, `congress109/phrases.txt` | congress109 phrase-count matrix (Taddy, *AoAS* 2015), snapshotted from R's `textir` package.                                                                                                                                                                                                                                       |

## Rebuilding the archive (maintainers only)

```bash
julia --project=. experiments/build-data-archive.jl
```

This fetches the inputs from their upstream mirrors, writes
`libsvm/manifest.json` and `MANIFEST.sha256`, and produces
`intercepts-data-v2.tar.gz` for upload to Zenodo.
