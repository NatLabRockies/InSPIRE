#!/bin/bash

CONF="$1"

if [[ $# -eq 0 ]] ; then
	echo 'no conf provided'
	exit 1
fi

states=(
  "alabama" "arizona" "arkansas" "california" "colorado" "connecticut"
  "delaware" "florida" "georgia" "idaho" "illinois" "indiana"
  "iowa" "kansas" "kentucky" "louisiana" "maine" "maryland"
  "massachusetts" "michigan" "minnesota" "mississippi" "missouri" "montana"
  "nebraska" "nevada" "new hampshire" "new jersey" "new mexico" "new york"
  "north carolina" "north dakota" "ohio" "oklahoma" "oregon" "pennsylvania"
  "rhode island" "south carolina" "south dakota" "tennessee" "texas" "utah"
  "vermont" "virginia" "washington" "west virginia" "wisconsin" "wyoming"
  "hawaii"
)

for state in "${states[@]}"
do
  echo "using ${state}"
  STATE_SLUG=${state// /_}   # replace spaces with underscores for job name

  sbatch --job-name=ag-irr-${STATE_SLUG}${CONF} scripts/submit_state_conf.sh "$state" "$CONF"
  echo "submitted for $state $CONF"
done
