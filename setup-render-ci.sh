#!/usr/bin/env bash

# Computo's render job never installs a Julia of its own, so `quarto render`
# would use whatever interpreter the runner image happens to ship (currently
# 1.12.6), and QuartoNotebookRunner rejects any Julia whose version differs from
# Manifest.toml's `julia_version`. Install exactly that version, instantiate the
# project against it, and point Quarto at it. The workflow file itself is
# journal-managed, so this hook is where such setup belongs.

set -euo pipefail

version=$(sed -n 's/^julia_version *= *"\(.*\)"/\1/p' Manifest.toml)

if [[ -z "$version" ]]; then
  echo "could not read julia_version from Manifest.toml" >&2
  exit 1
fi

prefix="$HOME/julia-$version"
url="https://julialang-s3.julialang.org/bin/linux/x64/${version%.*}/julia-$version-linux-x86_64.tar.gz"

echo "installing Julia $version from $url"

mkdir -p "$prefix"
curl -fsSL "$url" | tar -xz -C "$prefix" --strip-components=1

"$prefix/bin/julia" --version
"$prefix/bin/julia" --project=. -e 'using Pkg; Pkg.instantiate()'

echo "QUARTO_JULIA=$prefix/bin/julia" >>"$GITHUB_ENV"
