#!/bin/bash
# Discover nice shared color bounds for a metric, write bounds.env + shared colorbar.
# Usage (via sbatch from submit_publication_maps.sh):
#   sbatch viz/discover_crange.slurm <mode> <output_dir>
#
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nlr.gov
#SBATCH --mail-type=ALL

set -euo pipefail

MODE="${1:?mode is required}"
OUTPUT_DIR="${2:?output directory is required}"

PKG_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
VIZ_DIR="${PKG_ROOT}/viz"
cd "${VIZ_DIR}"
mkdir -p "${PKG_ROOT}/scripts/logs" "${OUTPUT_DIR}"

module load conda
source activate /projects/inspire/envs/render/
export SSL_CERT_FILE="$CONDA_PREFIX/ssl/cert.pem"
export REQUESTS_CA_BUNDLE="$CONDA_PREFIX/ssl/cert.pem"

PROBE_CONFIGS=(01 05 06 10 11)
DATA_DIR="/projects/inspire/PySAM-MAPS/v1.2/final-backup/"
BOUNDS_FILE="${OUTPUT_DIR}/bounds.env"
CMAP="inferno"
case "${MODE}" in
  shading-factor)
    CMAP="inferno_r"
    ;;
esac
SHARED_CBAR="${MODE}-${CMAP}-colorbar-fixed-crange.png"

CMD=(
  python generate_colorbars.py
  --mode "${MODE}"
  --configs "${PROBE_CONFIGS[@]}"
  --config-names publication-names
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --orientation vertical
  --aggregate-range
  --nice-bounds
  --cmap "${CMAP}"
  --bounds-file "${BOUNDS_FILE}"
  --shared-colorbar-name "${SHARED_CBAR}"
)

case "${MODE}" in
  annual-energy-per-acre)
    ;;
  mean-daily-insolation|shading-factor)
    # Probe winter/summer extremes only; maps still render annual + all 12 months.
    CMD+=(--months 1 6 7 12)
    ;;
  *)
    echo "Unsupported mode for discovery: ${MODE}" >&2
    exit 1
    ;;
esac

"${CMD[@]}"

echo "Discovery complete: ${BOUNDS_FILE}"
cat "${BOUNDS_FILE}"
