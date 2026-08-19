#!/usr/bin/env bash
# Orchestrate the skglm pre-fix vs post-fix controlled comparison
# (@fig-skglm-controlled in intercepts.qmd).
#
# Pipeline:
#   1. (re)generate the problem family via sim-skglm-controlled-problem.jl
#   2. for each source tree pinned by devenv.lock, run sim-skglm-controlled.py
#      with PYTHONPATH pointing at that tree
#
# Requires:
#   - the devenv shell, which exports both pinned source-tree paths

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

PRE_FIX_SHA="b03644fe"
POST_FIX_SHA="8f9cbd77"

RESULTS_DIR="$PROJECT_ROOT/results/skglm-controlled"
RESULTS_CSV="$RESULTS_DIR/results.csv"
INDEX_CSV="$RESULTS_DIR/index.csv"

for source_var in SKGLM_PRE_FIX_SOURCE SKGLM_POST_FIX_SOURCE; do
    source_dir="${!source_var:-}"
    if [[ ! -f "$source_dir/skglm/__init__.py" ]]; then
        echo "error: $source_var is unavailable; run this script inside 'devenv shell'" >&2
        exit 1
    fi
done

# 1. (re)generate the problem family if missing
if [[ ! -f "$INDEX_CSV" ]]; then
    echo "Generating problem family ..."
    julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/experiments/sim-skglm-controlled-problem.jl"
else
    echo "Using existing problem family at $INDEX_CSV"
fi

# 2. start with a fresh results.csv so the run is reproducible
rm -f "$RESULTS_CSV"

for spec in \
    "$PRE_FIX_SHA:$SKGLM_PRE_FIX_SOURCE" \
    "$POST_FIX_SHA:$SKGLM_POST_FIX_SOURCE"; do
    sha="${spec%%:*}"
    source_dir="${spec#*:}"
    echo
    echo "=== Running skglm at $sha ==="
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source_dir" \
        python "$PROJECT_ROOT/experiments/sim-skglm-controlled.py" \
        --commit "$sha" --out "$RESULTS_CSV" --index "$INDEX_CSV"
done

echo
echo "Done. Wrote $RESULTS_CSV"
wc -l "$RESULTS_CSV"
