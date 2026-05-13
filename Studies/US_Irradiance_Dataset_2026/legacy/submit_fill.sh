#!/bin/bash
#SBATCH --output=logs/agrivolt-irr-%x-%j.log
#SBATCH --error=logs/agrivolt-irr-%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17 # we changed this was previously 1, and changed from slurmcluster to localcluster
#SBATCH --mem=192G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

# jobname is defined in calling script (run_all_configs_state.sh)
STATE=$1
CONF=$2
GIDS=$3

LOGFILE=logs/agrivolt-irr-$SLURM_JOB_NAME-$SLURM_JOB_ID.log
ERRFILE=logs/agrivolt-irr-$SLURM_JOB_NAME-$SLURM_JOB_ID.err

module load anaconda3
source activate /home/tford/.conda-envs/geospatial

#### do work ####
agrivolt_ground_irradiance "$STATE" /projects/inspire/PySAM-MAPS/v0.5/"$STATE"-fill /home/tford/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM \
    --confs "$CONF" \
    --port 22118 \
    --workers 16 \
    --partition shared \
    --walltime 10:00:00 \
    --gids $GIDS

conda deactivate
