#!/bin/bash
#SBATCH --output=scripts/logs/agrivolt-irr-%x-%j.log
#SBATCH --error=scripts/logs/agrivolt-irr-%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17 # we changed this was previously 1, and changed from slurmcluster to localcluster
#SBATCH --mem=192G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

# DASK SCHEDULER OPTIONS
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.60
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.70
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95

# INTRA WORKER THREAD OPTIONS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

unset PANDAS_USE_PYARROW
unset PANDAS_STRING_STORAGE

# jobname is defined in calling script (run_all_configs_state.sh)
STATE=$1
CONF=$2

STATE_SLUG=${STATE// /_}

LOGFILE=scripts/logs/agrivolt-irr-$SLURM_JOB_NAME-$SLURM_JOB_ID.log
ERRFILE=scripts/logs/agrivolt-irr-$SLURM_JOB_NAME-$SLURM_JOB_ID.err

source scripts/_env_utils.sh
ENV_NAME="$(get_inspire_agrivolt_conda_env)"
CONF_DIR="$(get_inspire_agrivolt_SAM_conf_dir)"
OUTPUT_DIR="$(get_inspire_agrivolt_model_outs_dir)"

module load anaconda3
source activate "$ENV_NAME"

#### RUN SIMULATION USING CLI ####
agrivolt_ground_irradiance "$STATE" "$OUTPUT_DIR" "$CONF_DIR" --confs "$CONF" --port 22118

conda deactivate
