#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
conda activate /home/kdoubled/.conda-envs/s3env

srun python compare_datasets.py --data-file all_results.pkl --output comparison_results.csv

