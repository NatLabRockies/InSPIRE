#!/bin/bash
#SBATCH --output=scripts/logs/maps-%x-%j.out
#SBATCH --error=scripts/logs/maps-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=150G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

set -euo pipefail

MODE="${1:-mean-edgetoedge}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

cd "$REPO_ROOT"

source scripts/_env_utils.sh
ENV_NAME="$(get_inspire_agrivolt_conda_env)"

module load anaconda3
source activate "$ENV_NAME"

srun python viz/map.py --mode "$MODE"
