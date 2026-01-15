"""
Plot Wm2Front vs Distance Comparison
This script generates line plots comparing PySAM and bifacial radiance data
for Wm2Front vs distance (x) for each of the 11 setups (1-11).

The plot shows data from a combined pickle file (all_results.pkl) with a data_source column
indicating whether data is from 'bifacial_radiance' or 'pysam'.

Usage:
    python plot_wm2front_vs_distance.py --gid 886847 --timestamp "2023-01-01 12:00:00"
    python plot_wm2front_vs_distance.py --gid 886847 --timestamp "2023-06-21 12:00:00" --output June_21_comparison_plots.png
    python plot_wm2front_vs_distance.py --data-file all_results.pkl --gid 886847 --timestamp "2023-01-01 12:00:00"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import warnings


def plot_wm2front_vs_distance(
    data_file='all_results.pkl',
    gid=None,
    timestamp=None,
    output_file=None
):
    """
    Generate line plots comparing PySAM and bifacial radiance data for Wm2Front vs distance.
    
    Parameters
    ----------
    data_file : str
        Path to combined results pickle file with data_source column
    gid : int
        GID to plot (if None, uses first GID found)
    timestamp : str or pd.Timestamp, optional
        Timestamp to plot (e.g., "2023-01-01 12:00:00" or "2023-06-21 12:00:00")
        If None, uses first timestamp found in the data
    output_file : str, optional
        Output file path for the plot (default: wm2front_vs_distance_gid{gid}_{timestamp}.png)
    
    Returns
    -------
    str
        Path to saved plot file
    """
    print("Loading dataset...")
    
    # Load combined dataset
    all_data = pd.read_pickle(data_file)
    
    # Convert datetime columns
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    
    # Split into bifacial_radiance and pysam datasets
    validation = all_data[all_data['data_source'] == 'bifacial_radiance'].copy()
    pysam = all_data[all_data['data_source'] == 'pysam'].copy()
    
    # Select GID
    if gid is None:
        gid = validation['gid'].iloc[0]
        print(f"No GID specified, using first GID found: {gid}")
    else:
        print(f"Using GID: {gid}")
    
    # Check if GID exists in both datasets
    if gid not in validation['gid'].unique():
        raise ValueError(f"GID {gid} not found in validation data")
    if gid not in pysam['gid'].unique():
        raise ValueError(f"GID {gid} not found in pysam data")
    
    # Select timestamp
    if timestamp is None:
        target_datetime = validation[validation['gid'] == gid]['datetime'].iloc[0]
        print(f"No timestamp specified, using first timestamp found: {target_datetime}")
    else:
        # Parse timestamp string to datetime
        target_datetime = pd.to_datetime(timestamp)
        print(f"Using timestamp: {target_datetime}")
    
    # Normalize to remove seconds/microseconds for matching (keep only date and hour)
    target_datetime_normalized = target_datetime.replace(second=0, microsecond=0)
    
    # Normalize validation and pysam datetimes for matching (do this once before the loop)
    validation['datetime_normalized'] = validation['datetime'].apply(
        lambda dt: dt.replace(second=0, microsecond=0)
    )
    pysam['datetime_normalized'] = pysam['datetime'].apply(
        lambda dt: dt.replace(second=0, microsecond=0)
    )
    
    # Create figure with subplots for each setup (3 rows, 4 columns for 11 setups)
    # Share x and y axes among all subplots
    # Figure size: 6.5" width x 3" height
    fig, axes = plt.subplots(3, 4, figsize=(6.5, 3), sharey=True)
    
    # Format time for title (e.g., "9 am", "3 pm")
    hour = target_datetime.hour
    if hour == 0:
        time_str = "12 am"
    elif hour < 12:
        time_str = f"{hour} am"
    elif hour == 12:
        time_str = "12 pm"
    else:
        time_str = f"{hour - 12} pm"
    fig.suptitle(time_str, fontsize=10, fontweight='bold')
    
    axes = axes.flatten()
    
    # Collect all data to determine shared axis limits
    all_x_values = []
    all_y_values = []
    
    # Process each setup (1-11)
    for setup in range(1, 12):
        ax = axes[setup - 1]
        
        # Filter validation data for this GID, setup, and timestamp
        val_data = validation[
            (validation['gid'] == gid) &
            (validation['setup'] == setup) &
            (validation['datetime_normalized'] == target_datetime_normalized)
        ].copy()
        
        # Filter pysam data for this GID, setup, and timestamp
        pysam_data = pysam[
            (pysam['gid'] == gid) &
            (pysam['setup'] == setup) &
            (pysam['datetime_normalized'] == target_datetime_normalized)
        ].copy()
        
        if len(val_data) == 0 and len(pysam_data) == 0:
            ax.text(0.5, 0.5, f'Setup {setup}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{setup}', fontweight='bold', fontsize=8, pad=3)
            continue
        
        # Sort by x for plotting
        if len(val_data) > 0:
            val_data = val_data.sort_values('x')
            ax.plot(val_data['x'], val_data['Wm2Front'], 
                   '-', label='bifacial_radiance', linewidth=2, 
                   color="#0079C2", alpha=0.7)
            all_x_values.extend(val_data['x'].values)
            all_y_values.extend(val_data['Wm2Front'].values)
        
        if len(pysam_data) > 0:
            pysam_data = pysam_data.sort_values('x')
            ax.plot(pysam_data['x'], pysam_data['Wm2Front'], 
                   '-', label='SAM', linewidth=2, 
                   color="#F7A11A", alpha=0.7)
            all_x_values.extend(pysam_data['x'].values)
            all_y_values.extend(pysam_data['Wm2Front'].values)
        
        # Formatting
        ax.set_title(f'{setup}', fontweight='bold', fontsize=8, pad=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)  # Set tick label font size
        
        # Set x-axis limits for this subplot based on its own data
        if len(val_data) > 0 or len(pysam_data) > 0:
            subplot_x_values = []
            if len(val_data) > 0:
                subplot_x_values.extend(val_data['x'].values)
            if len(pysam_data) > 0:
                subplot_x_values.extend(pysam_data['x'].values)
            
            if len(subplot_x_values) > 0:
                subplot_x_min = min(subplot_x_values)
                subplot_x_max = max(subplot_x_values)
                subplot_x_range = subplot_x_max - subplot_x_min
                subplot_x_padding = 0.1 * subplot_x_range if subplot_x_range > 0 else 0.1
                ax.set_xlim(subplot_x_min - subplot_x_padding, subplot_x_max + subplot_x_padding)
                
                # Set x ticks at min and max with labels based on setup number
                ax.set_xticks([subplot_x_min - subplot_x_padding, subplot_x_max + subplot_x_padding])
                # Set labels: "W"/"E" for setups 1-5 and 11, "S"/"N" for setups 6-10
                if setup in [1, 2, 3, 4, 5, 11]:
                    ax.set_xticklabels(['W', 'E'])
                elif setup in [6, 7, 8, 9, 10]:
                    ax.set_xticklabels(['S', 'N'])
    
    # Set shared y-axis limits based on all data (x-axes are set per subplot)
    if len(all_y_values) > 0:
        y_min, y_max = 0, max(all_y_values)
        
        # Add some padding
        y_range = y_max - y_min
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        
        # Calculate final y limits
        y_limit_min = max(0, y_min - y_padding)
        y_limit_max = y_max + y_padding
        
        # Set y-axis limits on all axes (they're shared)
        for ax in axes[:11]:  # Only visible axes
            ax.set_ylim(y_limit_min, y_limit_max)
            # Set y-axis ticks to only show min (0) and hardcoded max (1000)
            ax.set_yticks([0, 1000])
    
    # Create a single shared legend for the entire figure
    # Get handles and labels from the first subplot that has data
    handles, labels = None, None
    for ax in axes[:11]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        if ax_handles:
            handles, labels = ax_handles, ax_labels
            break
    
    # Place legend in the 12th subplot (lower right corner)
    if handles and labels:
        # Turn off axes for the 12th subplot and use it for the legend
        axes[11].axis('off')
        # Place legend in the 12th subplot area (lower right corner)
        axes[11].legend(handles, labels, loc='center', fontsize=8, frameon=True)
    
    plt.tight_layout()
    # Adjust layout to make room for the x-axis label at the bottom and reduce vertical spacing
    plt.subplots_adjust(bottom=0.12, top=0.89, hspace=0.6)
    
    # Add single x and y axis labels to the figure (after tight_layout to position correctly)
    # Position x-label lower to avoid overlap with tick labels
    fig.supxlabel('Location within row-to-row pitch (m)', fontsize=12, y=-0.02)
    # Position y-label to the left to avoid overlap
    fig.supylabel('Ground Irradiance (W/m²)', fontsize=10, x=-0.02)
    
    # Save plot at 3" x 6" size (dpi will determine pixel resolution)
    if output_file is None:
        timestamp_filename = target_datetime_normalized.strftime('%Y-%m-%d_%H-%M')
        output_file = f'wm2front_vs_distance_gid{gid}_{timestamp_filename}.png'
    
    # Save the figure (size is already set to 3" x 6" in figsize)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    return output_file


def main():
    """Command-line interface for plotting Wm2Front vs distance."""
    parser = argparse.ArgumentParser(
        description='Plot Wm2Front vs Distance comparison for PySAM and validation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_wm2front_vs_distance.py --gid 886847 --timestamp "2023-01-01 12:00:00"
  python plot_wm2front_vs_distance.py --data-file all_results.pkl --gid 886847 --timestamp "2023-01-01 12:00:00"
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='all_results.pkl',
        help='Path to combined results pickle file with data_source column (default: all_results.pkl)'
    )
    
    parser.add_argument(
        '--gid',
        type=int,
        default=None,
        help='GID to plot (if not specified, uses first GID found)'
    )
    
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Timestamp to plot (e.g., "2023-01-01 12:00:00" or "2023-06-21 12:00:00"). If not specified, uses first timestamp found.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for the plot (default: wm2front_vs_distance_gid{gid}_{timestamp}.png)'
    )
    
    args = parser.parse_args()
    
    # Generate plot
    output_path = plot_wm2front_vs_distance(
        data_file=args.data_file,
        gid=args.gid,
        timestamp=args.timestamp,
        output_file=args.output
    )
    
    print(f"\nPlot generation complete!")
    return output_path


if __name__ == "__main__":
    main()

