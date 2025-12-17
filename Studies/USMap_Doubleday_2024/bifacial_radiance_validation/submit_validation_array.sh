#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --account=inspire
#SBATCH --array=0-164

# 0-164 (15 GIDs × 11 setups)

module load anaconda3
conda activate /home/kdoubled/.conda-envs/radianceEnv
module load gcc
source addPaths.sh

# Define the 15 GIDs
GIDS=(886847 243498 481324 852795 1116296 706260 478464 347412 1132667 138250 128689 981453 763236 1292659 191212)

# Calculate which GID and setup to use based on array task ID
# 165 jobs total: 15 GIDs × 11 setups
# Each GID gets 11 consecutive array indices (0-10, 11-21, etc.)
NUM_SETUPS=11
GID_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SETUPS))
SETUP=$((SLURM_ARRAY_TASK_ID % NUM_SETUPS + 1))

# Get the GID for this job
GID=${GIDS[$GID_INDEX]}

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "GID Index: $GID_INDEX"
echo "GID: $GID"
echo "Setup: $SETUP"

# Define results path and ensure it exists and is empty
RESULTS_PATH="/scratch/kdoubled/validation_results_v1"
# Only clear the directory on the first task to avoid race conditions
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    rm -rf "$RESULTS_PATH"
fi
mkdir -p "$RESULTS_PATH"

srun python bifacial_radiance_comparison.py --gid $GID --setup $SETUP --full_year --results_path "$RESULTS_PATH"

