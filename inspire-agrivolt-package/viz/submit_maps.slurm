#!/bin/bash
#SBATCH --output=scripts/logs/maps-%x-%j.out
#SBATCH --error=scripts/logs/maps-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=150G
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nlr.gov
#SBATCH --mail-type=ALL

set -euo pipefail

module load conda
source activate /projects/inspire/envs/render/
export SSL_CERT_FILE="$CONDA_PREFIX/ssl/cert.pem"
export REQUESTS_CA_BUNDLE="$CONDA_PREFIX/ssl/cert.pem"

for c in 01 02 03 04 05 06 07 08 09 10 11; do
  python map.py \
    --mode farmable-land-percent \
    --configs "$c" \
    --config-names publication-names \
    --cmin 40 --cmax 90
done