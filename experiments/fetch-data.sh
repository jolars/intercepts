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

STAGING="${TMP}/staging"
mkdir "${STAGING}"
DATA_FILES=()
ARCHIVE_FILES=()
while read -r _checksum path; do
  DATA_FILES+=("${path}")
  # Extract inputs supplied by the pinned archive.
  if [[ "${path}" != data/libsvm/w8a ]]; then
    ARCHIVE_FILES+=("${path}")
  fi
done < data/MANIFEST.sha256

# Extract only the committed inputs so archive metadata cannot replace the
# repository's checksum manifest or documentation.
echo "Unpacking into temporary staging directory"
tar -xzf "${TARBALL}" -C "${STAGING}" -- "${ARCHIVE_FILES[@]}"

if [[ " ${DATA_FILES[*]} " == *" data/libsvm/w8a "* ]]; then
  if tar -tzf "${TARBALL}" -- data/libsvm/w8a >/dev/null 2>&1; then
    tar -xzf "${TARBALL}" -C "${STAGING}" -- data/libsvm/w8a
  else
    echo "Retrieving w8a"
    W8A_SOURCE="${INTERCEPTS_W8A_SOURCE:-https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a}"
    mkdir -p "${STAGING}/data/libsvm"
    case "${W8A_SOURCE}" in
      http://* | https://*) curl -fL --retry 3 -o "${STAGING}/data/libsvm/w8a" "${W8A_SOURCE}" ;;
      *) cp "${W8A_SOURCE}" "${STAGING}/data/libsvm/w8a" ;;
    esac
  fi
fi

echo "Verifying checksums against data/MANIFEST.sha256"
(
  cd "${STAGING}"
  sha256sum -c "${REPO_ROOT}/data/MANIFEST.sha256"
)

# Preserve existing inputs until every file has passed verification.
for path in "${DATA_FILES[@]}"; do
  mkdir -p "$(dirname "${path}")"
  cp "${STAGING}/${path}" "${path}"
done

echo "Deriving Yeoh CSV inputs from the shipped RDS"
Rscript experiments/fetch-yeoh.R

echo "Done. Real-data inputs are staged under data/."
