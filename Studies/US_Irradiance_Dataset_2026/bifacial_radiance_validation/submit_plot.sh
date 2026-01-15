#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
conda activate /home/kdoubled/.conda-envs/s3env

# Create timestamped output directory
DIR_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="plots_${DIR_TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Default GID and data file
GID=886847
DATA_FILE="all_results.pkl"

# Generate heatmaps for setups 1, 6, 10, 11
# For both bifacial_radiance and pysam data sources
SETUPS=(1 6 10 11)

echo "Generating heatmaps..."

for SETUP in "${SETUPS[@]}"; do
    # Generate heatmap for bifacial_radiance
    echo "Generating heatmap for setup $SETUP, bifacial_radiance..."
    python plot_irradiance_heatmap.py \
        --data-file "$DATA_FILE" \
        --setup "$SETUP" \
        --gid "$GID" \
        --data-source bifacial_radiance \
        --output "$OUTPUT_DIR/setup_${SETUP}_heatmap_br.png" \
        --hour-start 5 --hour-end 18
    
    # Generate heatmap for pysam
    echo "Generating heatmap for setup $SETUP, pysam..."
    python plot_irradiance_heatmap.py \
        --data-file "$DATA_FILE" \
        --setup "$SETUP" \
        --gid "$GID" \
        --data-source pysam \
        --output "$OUTPUT_DIR/setup_${SETUP}_heatmap_pysam.png" \
        --hour-start 5 --hour-end 18
done

# Generate full resolution BR heatmaps
echo "Generating full resolution BR heatmaps..."
BR_FULL_RES_FILE="br_full_resolution_results.pkl"

if [ -f "$BR_FULL_RES_FILE" ]; then
    for SETUP in "${SETUPS[@]}"; do
        echo "Generating full resolution heatmap for setup $SETUP, bifacial_radiance..."
        python plot_irradiance_heatmap.py \
            --data-file "$BR_FULL_RES_FILE" \
            --setup "$SETUP" \
            --gid "$GID" \
            --data-source bifacial_radiance \
            --output "$OUTPUT_DIR/setup_${SETUP}_heatmap_br_full_res.png" \
            --hour-start 5 --hour-end 18
    done
else
    echo "Warning: $BR_FULL_RES_FILE not found. Skipping full resolution BR heatmaps."
fi

# Generate June 21 comparison plots for 9 am, noon, and 3 pm
echo "Generating June 21 comparison plots..."

TIMESTAMPS=(
    "2023-06-19 09:00:00"
    "2023-06-19 12:00:00"
    "2023-06-19 15:00:00"
)

for TIMESTAMP_STR in "${TIMESTAMPS[@]}"; do
    # Format timestamp for filename (replace spaces and colons)
    TIMESTAMP_FILENAME=$(echo "$TIMESTAMP_STR" | sed 's/ /_/g' | sed 's/://g')
    
    echo "Generating comparison plot for $TIMESTAMP_STR..."
    python plot_wm2front_vs_distance.py \
        --data-file "$DATA_FILE" \
        --gid "$GID" \
        --timestamp "$TIMESTAMP_STR" \
        --output "$OUTPUT_DIR/june21_${TIMESTAMP_FILENAME}_comparison.png"
done

echo "All plots generated in: $OUTPUT_DIR"

