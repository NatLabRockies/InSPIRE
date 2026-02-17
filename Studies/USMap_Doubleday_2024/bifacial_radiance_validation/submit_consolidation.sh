#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
conda activate /home/kdoubled/.conda-envs/s3env

srun python consolidate_all_results.py validation_results_v1 --base-path "/scratch/kdoubled/"

