#!/usr/bin/env bash
# =============================================================================
#  fetch_data.sh
#  Downloads two benchmark matrices from the SuiteSparse Matrix Collection
#  and extracts the .mtx files into the current directory.
#
#  Matrices:
#    thermomech_dK  – 204,316 × 204,316, ~2.9 M nnz  (regular, FEM)
#    ASIC_320k      – 321,671 × 321,671, ~1.3 M nnz  (highly irregular, EDA)
#
#  Usage:
#    chmod +x fetch_data.sh
#    ./fetch_data.sh
# =============================================================================

set -euo pipefail

# SuiteSparse base URL (HTTPS)
BASE="https://suitesparse-collection-website.herokuapp.com/MM"

# --------------------------------------------------------------------------
# fetch_matrix <group> <name>
#   Downloads <BASE>/<group>/<name>.tar.gz, extracts <name>/<name>.mtx,
#   moves the .mtx to the current directory, removes the tar and directory.
# --------------------------------------------------------------------------
fetch_matrix() {
    local group="$1"
    local name="$2"
    local archive="${name}.tar.gz"
    local url="${BASE}/${group}/${archive}"

    if [ -f "${name}.mtx" ]; then
        echo "[SKIP]  ${name}.mtx already exists."
        return
    fi

    echo "[DOWN]  Downloading ${name} from SuiteSparse..."
    wget --quiet --show-progress -O "${archive}" "${url}" || {
        echo "[ERR]   wget failed for ${url}"
        echo "        Try downloading manually and placing ${name}.mtx here."
        return 1
    }

    echo "[EXTR]  Extracting ${archive}..."
    tar -xzf "${archive}"

    # The tarball extracts to a sub-directory <name>/<name>.mtx
    if [ -f "${name}/${name}.mtx" ]; then
        mv "${name}/${name}.mtx" .
        rm -rf "${name}"
    else
        # Some tarballs put the .mtx directly at the root
        local found
        found=$(find . -maxdepth 2 -name "${name}.mtx" | head -n1)
        if [ -n "${found}" ] && [ "${found}" != "./${name}.mtx" ]; then
            mv "${found}" .
        fi
    fi

    rm -f "${archive}"
    echo "[OK]    ${name}.mtx ready."
}

# --------------------------------------------------------------------------
# Check dependencies
# --------------------------------------------------------------------------
for cmd in wget tar; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "ERROR: '${cmd}' not found. Please install it and re-run."
        exit 1
    fi
done

# --------------------------------------------------------------------------
# Fetch both matrices
# --------------------------------------------------------------------------
echo "======================================================="
echo " SuiteSparse Matrix Downloader"
echo " Target directory: $(pwd)"
echo "======================================================="

fetch_matrix "Botonakis"  "thermomech_dK"
fetch_matrix "Sandia"     "ASIC_320k"

echo ""
echo "======================================================="
echo " Done. Files present:"
ls -lh ./*.mtx 2>/dev/null || echo "  (none found – check errors above)"
echo ""
echo " Quick build & run:"
echo "   make"
echo "   ./spmv_benchmark --mtx=thermomech_dK.mtx"
echo "   ./spmv_benchmark --mtx=ASIC_320k.mtx"
echo "======================================================="
