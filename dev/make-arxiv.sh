#!/usr/bin/env bash
# Build an arXiv-ready source tarball from the rendered paper.
#
# Pipeline:
#   1. (re)render intercepts.qmd to computo-pdf if intercepts.tex is stale,
#      which also refreshes intercepts_files/figure-pdf/
#   2. stage the .tex plus only the graphics it actually references
#   3. strip the journal-review furniture from the staged copy: the "submitted"
#      watermark and the line numbers. The watermark is a claim about the
#      paper's status at Computo and has no business on a preprint; the line
#      numbers exist for referees.
#   4. test-compile the staged tree and refuse to package anything that does
#      not build cleanly or that still carries the watermark
#   5. tar it up
#
# The Computo extension regenerates intercepts.tex on every render, so step 3
# edits the staged copy only --- never the working tree.
#
# Requires: quarto, lualatex. Uses pdftotext for the watermark check when
# available.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

PAPER="intercepts"
# Declared in the staged source and used for the verification build, so the two
# cannot drift apart.
ENGINE="lualatex"
SRC_TEX="$PROJECT_ROOT/$PAPER.tex"
SRC_QMD="$PROJECT_ROOT/$PAPER.qmd"
STAGE="$PROJECT_ROOT/.arxiv"
TARBALL="$PROJECT_ROOT/$PAPER-arxiv.tar.gz"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# 1. Render only if needed --- a full render costs minutes.
if [[ "${ARXIV_SKIP_RENDER:-0}" == 1 ]]; then
    [[ -f "$SRC_TEX" ]] || die "ARXIV_SKIP_RENDER=1 but $PAPER.tex does not exist"
    log "skipping render (ARXIV_SKIP_RENDER=1)"
elif [[ ! -f "$SRC_TEX" || "$SRC_QMD" -nt "$SRC_TEX" ]]; then
    log "rendering $PAPER.qmd to computo-pdf"
    (cd "$PROJECT_ROOT" && quarto render "$PAPER.qmd" --to computo-pdf)
else
    log "$PAPER.tex is up to date"
fi

# 2. Stage the source.
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$SRC_TEX" "$STAGE/$PAPER.tex"

# Copy every graphic the document references, preserving relative paths, so a
# change of figure format or a branded header that pulls in a logo is picked up
# without editing this script.
mapfile -t figures < <(
    grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' "$SRC_TEX" |
        sed 's/.*{//;s/}//' | sort -u
)
[[ ${#figures[@]} -gt 0 ]] || die "no \\includegraphics targets found in $PAPER.tex"

for fig in "${figures[@]}"; do
    [[ -f "$PROJECT_ROOT/$fig" ]] || die "referenced graphic is missing: $fig"
    mkdir -p "$STAGE/$(dirname "$fig")"
    cp "$PROJECT_ROOT/$fig" "$STAGE/$fig"
done
log "staged $PAPER.tex and ${#figures[@]} graphics"

# 3. Strip the review furniture. Both patterns are asserted rather than applied
# best-effort: if a future extension release renames them, this should fail
# loudly instead of silently shipping a watermarked preprint.
watermarks=$(grep -c '^\\DraftwatermarkOptions{stamp=true' "$STAGE/$PAPER.tex" || true)
linenos=$(grep -c '^\\linenumbers$' "$STAGE/$PAPER.tex" || true)

[[ "$watermarks" -ge 1 ]] ||
    die "no active watermark found --- has the Computo extension changed how it stamps drafts?"
[[ "$linenos" -ge 1 ]] ||
    die "no \\linenumbers found --- has the Computo extension changed its review layout?"

sed -i \
    -e 's/^\\DraftwatermarkOptions{stamp=true.*$/\\DraftwatermarkOptions{stamp=false}/' \
    -e '/^\\linenumbers$/d' \
    "$STAGE/$PAPER.tex"
log "stripped watermark ($watermarks) and line numbers ($linenos)"

# The document needs LuaLaTeX --- unicode-math, fontspec and libertinus rule out
# pdflatex. This magic comment is the editor convention (TeXShop, TeXworks,
# VS Code LaTeX Workshop); arXiv's own lever is a 00README.XXX, so treat this as
# documentation for whoever opens the source rather than as a guarantee that a
# build service will honour it. It is a plain comment, so it is free either way.
sed -i "1i %!TEX program = $ENGINE" "$STAGE/$PAPER.tex"
log "declared engine: $ENGINE"

# 4. Prove the staged tree builds on its own, the way arXiv will build it.
if [[ "${ARXIV_SKIP_VERIFY:-0}" == 1 ]]; then
    log "skipping verification build (ARXIV_SKIP_VERIFY=1)"
else
    command -v "$ENGINE" >/dev/null || die "$ENGINE not found; set ARXIV_SKIP_VERIFY=1 to package anyway"
    log "test-compiling the staged source with $ENGINE"
    (
        cd "$STAGE"
        for pass in 1 2 3; do
            "$ENGINE" -interaction=nonstopmode "$PAPER.tex" >"pass$pass.log" 2>&1 ||
                { tail -40 "pass$pass.log" >&2; die "$ENGINE failed on pass $pass"; }
        done

        errors=$(grep -cE '^! ' pass3.log || true)
        [[ "$errors" -eq 0 ]] || { grep -E '^! ' pass3.log >&2; die "$errors LaTeX error(s)"; }

        ! grep -q 'There were undefined references' pass3.log ||
            die "undefined references remain after three passes"

        if command -v pdftotext >/dev/null; then
            stamped=$(pdftotext "$PAPER.pdf" - 2>/dev/null | grep -ci submitted || true)
            [[ "$stamped" -eq 0 ]] || die "the built PDF still contains 'submitted' ($stamped hits)"
        fi

        if command -v pdfinfo >/dev/null; then
            log "built $(pdfinfo "$PAPER.pdf" | awk '/^Pages/ {print $2}') pages, no errors"
        fi
    )
    # Build products are not part of the submission.
    rm -f "$STAGE"/*.log "$STAGE"/*.aux "$STAGE"/*.out "$STAGE"/*.toc "$STAGE/$PAPER.pdf"
fi

# 5. Package.
rm -f "$TARBALL"
tar czf "$TARBALL" -C "$STAGE" .
log "wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"
