#!/bin/bash
# Submit discover + dependent render jobs for a publication map metric.
#
# Usage (from inspire-agrivolt-package or viz/):
#   ./viz/submit_publication_maps.sh annual-energy-per-acre
#   ./viz/submit_publication_maps.sh mean-daily-insolation
#   ./viz/submit_publication_maps.sh shading-factor
#
# Optional second argument overrides the output directory.

set -euo pipefail

MODE="${1:?Usage: $0 <annual-energy-per-acre|mean-daily-insolation|shading-factor> [output_dir]}"

VIZ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${VIZ_DIR}/.." && pwd)"
cd "${PKG_ROOT}"
mkdir -p scripts/logs

case "${MODE}" in
  annual-energy-per-acre)
    DEFAULT_OUT_DIR="${VIZ_DIR}/maps/annual_energy_per_acre"
    ARRAY_SPEC="0"
    ;;
  mean-daily-insolation)
    DEFAULT_OUT_DIR="${VIZ_DIR}/maps/mean_daily_insolation"
    ARRAY_SPEC="0-12"
    ;;
  shading-factor)
    DEFAULT_OUT_DIR="${VIZ_DIR}/maps/shading_factor"
    ARRAY_SPEC="0-12"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    echo "Expected one of: annual-energy-per-acre, mean-daily-insolation, shading-factor" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${2:-${DEFAULT_OUT_DIR}}"
mkdir -p "${OUTPUT_DIR}"

JOB_STEM="${MODE}"
DISCOVER_NAME="maps-discover-${JOB_STEM}"
RENDER_NAME="maps-render-${JOB_STEM}"

echo "Submitting discover job for ${MODE}"
DISCOVER_JOB_ID="$(
  sbatch --parsable \
    --job-name="${DISCOVER_NAME}" \
    "${VIZ_DIR}/discover_crange.slurm" \
    "${MODE}" \
    "${OUTPUT_DIR}"
)"
echo "Discover job id: ${DISCOVER_JOB_ID}"

echo "Submitting render array ${ARRAY_SPEC} dependent on ${DISCOVER_JOB_ID}"
RENDER_JOB_ID="$(
  sbatch --parsable \
    --job-name="${RENDER_NAME}" \
    --dependency="afterok:${DISCOVER_JOB_ID}" \
    --array="${ARRAY_SPEC}" \
    "${VIZ_DIR}/render_maps.slurm" \
    "${MODE}" \
    "${OUTPUT_DIR}"
)"
echo "Render job id: ${RENDER_JOB_ID}"
echo "Outputs will be written to: ${OUTPUT_DIR}"
