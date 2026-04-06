#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_CONFIG="${1:-1}"
STOP_CONFIG="${2:-11}"

cd "$REPO_ROOT"

if (( START_CONFIG > STOP_CONFIG )); then
  echo "start config must be less than or equal to stop config" >&2
  exit 1
fi

for ((config=START_CONFIG; config<=STOP_CONFIG; config++)); do
  job_name=$(printf "map-conf%02d" "$config")
  echo "Submitting ${job_name}"
  sbatch --job-name="$job_name" viz/submit_map.slurm "$config"
done
