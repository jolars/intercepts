#!/usr/bin/env bash
#
# Retrieve the real-data inputs for the paper's real-data experiments from the
# pinned Zenodo archive, verify them against the committed checksum manifest,
# and stage them under data/. You only need this to RE-RUN the real-data
# experiments; the cached results under results/ render the paper without it.
#
#   bash experiments/fetch-data.sh
#
# To test against a locally built archive before the Zenodo DOI is minted (see
# experiments/build-data-archive.jl), point INTERCEPTS_DATA_ARCHIVE at it:
#
#   INTERCEPTS_DATA_ARCHIVE=./intercepts-data-v1.tar.gz bash experiments/fetch-data.sh

set -euo pipefail

ARCHIVE_VERSION="v1"
ARCHIVE_NAME="intercepts-data-${ARCHIVE_VERSION}.tar.gz"

# Pinned Zenodo deposit of ${ARCHIVE_NAME} (DOI 10.5281/zenodo.20315625).
ZENODO_RECORD="20315625"
ZENODO_DOI="10.5281/zenodo.${ZENODO_RECORD}"
ARCHIVE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE="${INTERCEPTS_DATA_ARCHIVE:-${ARCHIVE_URL}}"

if [[ "${SOURCE}" == *XXXXXXX* ]]; then
  echo "error: the Zenodo record id is still a placeholder in $0." >&2
  echo "       Fill in ZENODO_RECORD/ZENODO_DOI after uploading ${ARCHIVE_NAME}," >&2
  echo "       or set INTERCEPTS_DATA_ARCHIVE to a local copy for testing." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
TARBALL="${TMP}/${ARCHIVE_NAME}"

echo "Retrieving real-data inputs (DOI ${ZENODO_DOI})"
echo "  source: ${SOURCE}"
case "${SOURCE}" in
  http://* | https://*)
    curl -fL --retry 3 -o "${TARBALL}" "${SOURCE}"
    ;;
  *)
    cp "${SOURCE}" "${TARBALL}"
    ;;
esac

echo "Unpacking into ${REPO_ROOT}"
tar -xzf "${TARBALL}" -C "${REPO_ROOT}"

echo "Verifying checksums against data/MANIFEST.sha256"
sha256sum -c data/MANIFEST.sha256

echo "Deriving Yeoh CSV inputs from the shipped RDS"
Rscript experiments/fetch-yeoh.R

echo "Done. Real-data inputs are staged under data/."
