#!/bin/bash
# Render fixed-crange maps for one period (array task) across all 11 configs.
# Array index: 0 = annual, 1-12 = month. For annual-only metrics, use --array=0.
#
# Usage:
#   sbatch --array=0-12 viz/render_maps.slurm <mode> <output_dir>
#   sbatch --array=0 viz/render_maps.slurm annual-energy-per-acre <output_dir>
#
#SBATCH --output=scripts/logs/maps-render-%x-%A_%a.out
#SBATCH --error=scripts/logs/maps-render-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=150G
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nlr.gov
#SBATCH --mail-type=ALL

set -euo pipefail

MODE="${1:?mode is required}"
OUTPUT_DIR="${2:?output directory is required}"

VIZ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${VIZ_DIR}/.." && pwd)"
cd "${VIZ_DIR}"
mkdir -p "${PKG_ROOT}/scripts/logs" "${OUTPUT_DIR}"

BOUNDS_FILE="${OUTPUT_DIR}/bounds.env"
if [[ ! -f "${BOUNDS_FILE}" ]]; then
  echo "Missing bounds file: ${BOUNDS_FILE}" >&2
  echo "Run discover_crange.slurm first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${BOUNDS_FILE}"

module load conda
source activate /projects/inspire/envs/render/
export SSL_CERT_FILE="$CONDA_PREFIX/ssl/cert.pem"
export REQUESTS_CA_BUNDLE="$CONDA_PREFIX/ssl/cert.pem"

PERIOD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

case "${MODE}" in
  annual-energy-per-acre)
    if [[ "${PERIOD_INDEX}" -ne 0 ]]; then
      echo "annual-energy-per-acre only supports array task 0 (annual)." >&2
      exit 1
    fi
    ;;
  mean-daily-insolation|shading-factor)
    if [[ "${PERIOD_INDEX}" -lt 0 || "${PERIOD_INDEX}" -gt 12 ]]; then
      echo "Invalid array task id ${PERIOD_INDEX}; expected 0-12." >&2
      exit 1
    fi
    ;;
  *)
    echo "Unsupported mode for render: ${MODE}" >&2
    exit 1
    ;;
esac

for c in 01 02 03 04 05 06 07 08 09 10 11; do
  CMD=(
    python map.py
    --mode "${MODE}"
    --configs "${c}"
    --config-names publication-names
    --cmin "${CMIN}"
    --cmax "${CMAX}"
    --output-dir "${OUTPUT_DIR}"
  )
  if [[ "${MODE}" == "mean-daily-insolation" || "${MODE}" == "shading-factor" ]]; then
    if [[ "${PERIOD_INDEX}" -ge 1 ]]; then
      CMD+=(--month "${PERIOD_INDEX}")
    fi
  fi
  "${CMD[@]}"
done
