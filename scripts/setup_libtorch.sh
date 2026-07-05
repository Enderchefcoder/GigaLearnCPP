#!/usr/bin/env bash
# Downloads libtorch and extracts it to GigaLearnCPP/libtorch (where the build expects it).
#
# Usage:
#   ./scripts/setup_libtorch.sh            # CPU-only build
#   ./scripts/setup_libtorch.sh cu124      # CUDA 12.4 build
#   ./scripts/setup_libtorch.sh cu121      # CUDA 12.1 build
#   LIBTORCH_VERSION=2.5.1 ./scripts/setup_libtorch.sh   # Override the version
set -euo pipefail

FLAVOR="${1:-cpu}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.5.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/../GigaLearnCPP"

if [ -d "$DEST_DIR/libtorch" ]; then
	echo "libtorch already exists at $DEST_DIR/libtorch"
	echo "Delete that folder first if you want to reinstall."
	exit 0
fi

URL="https://download.pytorch.org/libtorch/${FLAVOR}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${FLAVOR}.zip"

echo "Downloading libtorch ${LIBTORCH_VERSION} (${FLAVOR}) ..."
echo "  ${URL}"

TMP_ZIP="$(mktemp --suffix=.zip)"
trap 'rm -f "$TMP_ZIP"' EXIT

curl -fL --progress-bar -o "$TMP_ZIP" "$URL" || {
	echo "Download failed. Check that the version/flavor combination exists at https://pytorch.org/get-started/locally/"
	exit 1
}

echo "Extracting to $DEST_DIR/libtorch ..."
unzip -q "$TMP_ZIP" -d "$DEST_DIR"

echo "Done. Configure the project with:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
