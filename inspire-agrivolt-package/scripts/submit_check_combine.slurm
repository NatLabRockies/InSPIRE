#!/bin/bash
#SBATCH --output=scripts/logs/agrivolt-irr-combine-check-%x-%j.out
#SBATCH --error=scripts/logs/agrivolt-irr-combine-check-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=150G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

source scripts/_env_utils.sh
ENV_NAME="$(get_inspire_agrivolt_conda_env)"
FINAL_DIR="$(get_inspire_agrivolt_final_dir)"

module load anaconda3
source activate "$ENV_NAME"

export PYTHONUNBUFFERED=1
srun -u python scripts/check_combine.py $FINAL_DIR
