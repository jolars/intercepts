#!/usr/bin/env bash
# Compare correctly scaled gradient and Newton intercept updates in skglm
# (@fig-skglm-controlled in main.qmd).
#
# Pipeline:
#   1. (re)generate the problem family via sim-skglm-controlled-problem.jl
#   2. run both strategies against the same source tree pinned by devenv.lock
#
# Requires:
#   - the devenv shell, which exports SKGLM_POST_FIX_SOURCE

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

RESULTS_DIR="$PROJECT_ROOT/results/skglm-controlled"
RESULTS_CSV="$RESULTS_DIR/results.csv"
INDEX_CSV="$RESULTS_DIR/index.csv"

source_dir="${SKGLM_POST_FIX_SOURCE:-}"
if [[ ! -f "$source_dir/skglm/__init__.py" ]]; then
    echo "error: SKGLM_POST_FIX_SOURCE is unavailable; run inside 'devenv shell'" >&2
    exit 1
fi

problem_family_complete() {
    python - "$INDEX_CSV" <<'PY'
import csv
from pathlib import Path
import sys

index = Path(sys.argv[1])
if not index.is_file():
    sys.exit(1)
with index.open(newline="") as stream:
    reader = csv.DictReader(stream)
    if not reader.fieldnames or "cell_name" not in reader.fieldnames:
        sys.exit(1)
    rows = list(reader)
paths = [
    index.parent / "cells" / row["cell_name"] / name
    for row in rows
    for name in ("X.csv", "y.csv", "meta.json")
]
sys.exit(0 if rows and all(p.is_file() and p.stat().st_size for p in paths) else 1)
PY
}

# The committed index can survive a checkout without its ignored cell inputs.
if ! problem_family_complete; then
    echo "Generating problem family ..."
    julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/experiments/sim-skglm-controlled-problem.jl"
else
    echo "Using existing problem family at $INDEX_CSV"
fi

if ! problem_family_complete; then
    echo "error: problem generation left missing or empty cell inputs at $INDEX_CSV" >&2
    exit 1
fi

# Replace the committed cache only after the comparison completes successfully.
skglm_results_tmp="$(mktemp "$RESULTS_DIR/results.XXXXXX.csv")"
trap 'rm -f -- "$skglm_results_tmp"' EXIT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source_dir" \
    python "$PROJECT_ROOT/experiments/sim-skglm-controlled.py" \
    --out "$skglm_results_tmp" --index "$INDEX_CSV"
mv -- "$skglm_results_tmp" "$RESULTS_CSV"

echo
echo "Done. Wrote $RESULTS_CSV"
wc -l "$RESULTS_CSV"
