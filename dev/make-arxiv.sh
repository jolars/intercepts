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
# Requires: quarto, latexmk, lualatex. Uses pdftotext for the watermark check
# when available.

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

resolve_graphic() {
    local fig="$1"
    local extension

    if [[ -f "$PROJECT_ROOT/$fig" ]]; then
        printf '%s\n' "$fig"
        return
    fi

    # LaTeX permits omitted extensions in \includegraphics; the archive must
    # contain the concrete file that graphicx resolves during compilation.
    for extension in .pdf .png .jpg .jpeg .eps; do
        if [[ -f "$PROJECT_ROOT/$fig$extension" ]]; then
            printf '%s\n' "$fig$extension"
            return
        fi
    done

    return 1
}

for fig in "${figures[@]}"; do
    resolved=$(resolve_graphic "$fig") || die "referenced graphic is missing: $fig"
    mkdir -p "$STAGE/$(dirname "$resolved")"
    cp "$PROJECT_ROOT/$resolved" "$STAGE/$resolved"
done
log "staged $PAPER.tex and ${#figures[@]} graphics"

# 3. Strip the review furniture when the rendered format enabled it. The
# verification below independently rejects a PDF that still contains the
# submitted watermark.
watermarks=$(grep -c '^\\DraftwatermarkOptions{stamp=true' "$STAGE/$PAPER.tex" || true)
linenos=$(grep -c '^\\linenumbers$' "$STAGE/$PAPER.tex" || true)

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
    command -v latexmk >/dev/null || die "latexmk not found; set ARXIV_SKIP_VERIFY=1 to package anyway"
    log "test-compiling the staged source with $ENGINE"
    (
        cd "$STAGE"
        latexmk -lualatex -interaction=nonstopmode -halt-on-error "$PAPER.tex" >build.log 2>&1 ||
            { tail -40 build.log >&2; die "latexmk failed"; }

        errors=$(grep -cE '^! ' "$PAPER.log" || true)
        [[ "$errors" -eq 0 ]] || { grep -E '^! ' "$PAPER.log" >&2; die "$errors LaTeX error(s)"; }

        ! grep -q 'There were undefined references' "$PAPER.log" ||
            die "undefined references remain after latexmk"

        if command -v pdftotext >/dev/null; then
            stamped=$(pdftotext "$PAPER.pdf" - 2>/dev/null | grep -ci submitted || true)
            [[ "$stamped" -eq 0 ]] || die "the built PDF still contains 'submitted' ($stamped hits)"
        fi

        if command -v pdfinfo >/dev/null; then
            log "built $(pdfinfo "$PAPER.pdf" | awk '/^Pages/ {print $2}') pages, no errors"
        fi
    )
    # Build products are not part of the submission.
    rm -f \
        "$STAGE"/*.log \
        "$STAGE"/*.aux \
        "$STAGE"/*.out \
        "$STAGE"/*.toc \
        "$STAGE"/*.fls \
        "$STAGE"/*.fdb_latexmk \
        "$STAGE/$PAPER.pdf"
fi

# 5. Package.
rm -f "$TARBALL"
tar czf "$TARBALL" -C "$STAGE" .
log "wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"
