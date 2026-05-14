#!/usr/bin/env bash
# Orchestrate the skglm pre-fix vs post-fix controlled comparison
# (@fig-skglm-controlled in intercepts.qmd).
#
# Pipeline:
#   1. (re)generate the problem family via sim-skglm-controlled-problem.jl
#   2. for each pinned commit of the local skglm checkout, run
#      sim-skglm-controlled.py with PYTHONPATH pointing at the checkout so
#      the run uses that exact source tree
#
# Requires:
#   - a clean working tree at $SKGLM_REPO (otherwise we refuse to checkout
#     over the user's uncommitted changes)
#   - both pinned SHAs reachable in that repo

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SKGLM_REPO="${SKGLM_REPO:-/home/jola/projects/skglm}"

PRE_FIX_SHA="b03644fe"
POST_FIX_SHA="8f9cbd77"

RESULTS_DIR="$PROJECT_ROOT/results/skglm-controlled"
RESULTS_CSV="$RESULTS_DIR/results.csv"
INDEX_CSV="$RESULTS_DIR/index.csv"

if [[ ! -d "$SKGLM_REPO/.git" ]]; then
    echo "error: SKGLM_REPO=$SKGLM_REPO is not a git checkout" >&2
    exit 1
fi

if [[ -n "$(git -C "$SKGLM_REPO" status --porcelain)" ]]; then
    echo "error: $SKGLM_REPO has uncommitted changes; aborting before checkout" >&2
    git -C "$SKGLM_REPO" status --short >&2
    exit 1
fi

original_ref="$(git -C "$SKGLM_REPO" rev-parse --abbrev-ref HEAD)"
if [[ "$original_ref" == "HEAD" ]]; then
    original_ref="$(git -C "$SKGLM_REPO" rev-parse HEAD)"
fi
echo "Will restore $SKGLM_REPO to '$original_ref' on exit."

restore_skglm() {
    git -C "$SKGLM_REPO" checkout "$original_ref" -- skglm/ 2>/dev/null || true
}
trap restore_skglm EXIT

for sha in "$PRE_FIX_SHA" "$POST_FIX_SHA"; do
    if ! git -C "$SKGLM_REPO" cat-file -e "$sha^{commit}" 2>/dev/null; then
        echo "error: commit $sha not found in $SKGLM_REPO" >&2
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

for sha in "$PRE_FIX_SHA" "$POST_FIX_SHA"; do
    echo
    echo "=== Checking out skglm at $sha ==="
    git -C "$SKGLM_REPO" checkout "$sha" -- skglm/
    # Drop bytecode caches in case stale .pyc shadow source differences
    find "$SKGLM_REPO/skglm" -type d -name __pycache__ -prune -exec rm -rf {} +

    PYTHONPATH="$SKGLM_REPO" python "$PROJECT_ROOT/experiments/sim-skglm-controlled.py" \
        --commit "$sha" --out "$RESULTS_CSV" --index "$INDEX_CSV"
done

echo
echo "Done. Wrote $RESULTS_CSV"
wc -l "$RESULTS_CSV"
