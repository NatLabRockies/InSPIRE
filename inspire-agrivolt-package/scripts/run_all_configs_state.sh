#!/bin/bash

configs=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10", "11")

STATE="$1"

if [[ $# -eq 0 ]] ; then
	echo 'no state provided'
	exit 1
fi

echo "using ${STATE}"
STATE_SLUG=${STATE// /_}   # replace spaces with underscores for job name

for conf in "${configs[@]}"
do
  sbatch --job-name=ag-irr-${STATE_SLUG}${conf} submit_state_conf.sh "$STATE" "$conf"
  echo "submitted for $STATE_SLUG $conf"
done
