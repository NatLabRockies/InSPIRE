#!/bin/bash
#SBATCH --output=scripts/logs/agrivolt-irr-postprocess-%x-%j.out
#SBATCH --error=scripts/logs/agrivolt-irr-postprocess-%x-%j.err
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
MODEL_OUTS_DIR="$(get_inspire_agrivolt_model_outs_dir)"
POSTPROCESS_DIR="$(get_inspire_agrivolt_postprocess_dir)"

module load anaconda3
source activate "$ENV_NAME"

srun python scripts/postprocess.py $MODEL_OUTS_DIR $POSTPROCESS_DIR