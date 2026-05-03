#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-mean-edgetoedge}"

cd "$REPO_ROOT"

job_name=$(printf "map-%s" "$MODE")
echo "Submitting ${job_name}"
sbatch --job-name="$job_name" viz/submit_map.slurm "$MODE"
