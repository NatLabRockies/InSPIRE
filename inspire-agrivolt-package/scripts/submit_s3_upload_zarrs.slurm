#!/bin/bash
#SBATCH --output=scripts/logs/agrivolt-irr-deploy-%x-%j.out
#SBATCH --error=scripts/logs/agrivolt-irr-deploy-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=shared
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

set -euo pipefail

source scripts/_env_utils.sh
ENV_NAME="$(get_inspire_agrivolt_conda_env)"
FINAL_DIR="$(get_inspire_agrivolt_final_dir)"

module load anaconda3
source activate "$ENV_NAME"

zarr_names=("01.zarr" "02.zarr" "03.zarr" "04.zarr" "05.zarr" "06.zarr" "07.zarr" "08.zarr" "09.zarr" "10.zarr" "11.zarr")

for local_zarr_name in "${zarr_names[@]}"; do
    srun --exclusive -N1 -n1 python scripts/s3_upload.py "$local_zarr_name" $FINAL_DIR --rotate-prefix &
done

wait
