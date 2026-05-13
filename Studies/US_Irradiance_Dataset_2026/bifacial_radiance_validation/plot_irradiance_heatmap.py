"""
Plot Irradiance Heatmap from Pickle File
This script generates heatmaps of Wm2Front (irradiance) values from all_results.pkl or 
br_full_resolution_results.pkl DataFrame, aggregated by hour of day or showing all time points.

Usage:
    python plot_irradiance_heatmap.py --setup 1 --gid 886847 --data-source bifacial_radiance --output heatmap.png
    python plot_irradiance_heatmap.py --setup 1 --gid 886847 --data-file br_full_resolution_results.pkl --output heatmap.png
"""

import pandas as pd
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_irradiance_heatmap_from_pkl(df, setup, gid, data_source=None, output_file=None,
                                      aggregate_by_hour=True, aggregation='mean',
                                      figsize=(12, 8), cmap='viridis',
                                      hour_start=None, hour_end=None):
    """
    Plot a heatmap of Wm2Front (irradiance) values from all_results.pkl or br_full_resolution_results.pkl DataFrame,
    aggregated by hour of day.
    
    Works with both aggregated (10 distance points) and full resolution (100 distance points) datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from pickle file with columns: gid, setup, datetime, x, Wm2Front, data_source
    setup : int
        Setup number to plot
    gid : int
        GID to plot
    data_source : str, optional
        Filter by data_source ('bifacial_radiance' or 'pysam'). If None, uses all data sources.
    output_file : str, optional
        Path to save the plot. If None, displays the plot.
    aggregate_by_hour : bool, default True
        If True, aggregate across all days for each hour to create a heatmap.
        If False, show all time points on y-axis.
    aggregation : str, default 'mean'
        Aggregation method when aggregate_by_hour=True. Options: 'mean', 'max', 'min', 'sum'
    hour_start : int, optional
        Start hour of day (0-23) for y-axis range. If None, uses hours with data.
    hour_end : int, optional
        End hour of day (0-23, inclusive) for y-axis range. If None, uses hours with data.
    figsize : tuple, default (12, 8)
        Figure size (width, height) in inches
    cmap : str, default 'viridis'
        Colormap to use for the heatmap
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Filter data by setup and gid
    filtered_df = df[(df['setup'] == setup) & (df['gid'] == gid)].copy()
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for setup {setup} and GID {gid}")
    
    # Filter by data_source if specified
    if data_source is not None:
        filtered_df = filtered_df[filtered_df['data_source'] == data_source]
        if len(filtered_df) == 0:
            raise ValueError(f"No data found for setup {setup}, GID {gid}, and data_source {data_source}")
    
    # Ensure datetime is datetime type
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['datetime']):
        filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
    
    # Get unique distances and times
    distances = sorted(filtered_df['x'].unique())
    times = sorted(filtered_df['datetime'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if aggregate_by_hour:
        # Extract hour of day
        filtered_df['hour'] = filtered_df['datetime'].dt.hour
        
        # Determine which hours to include on y-axis
        if hour_start is not None and hour_end is not None:
            # Use specified hour range
            if hour_start < 0 or hour_start > 23 or hour_end < 0 or hour_end > 23:
                raise ValueError("hour_start and hour_end must be between 0 and 23")
            if hour_start > hour_end:
                raise ValueError("hour_start must be <= hour_end")
            hours_to_show = list(range(hour_start, hour_end + 1))
        else:
            # Fall back to detecting hours with data
            hours_to_show = sorted(filtered_df['hour'].unique())
            if len(hours_to_show) == 0:
                raise ValueError("No data found for any hour of day")
        
        # Initialize array for aggregated data: hours to show x distances
        data_by_hour = np.full((len(hours_to_show), len(distances)), np.nan)
        
        # Aggregate data for each hour in the range
        for hour_idx, hour in enumerate(hours_to_show):
            hour_data = filtered_df[filtered_df['hour'] == hour]
            
            # Pivot to get distance x time structure, then aggregate
            for dist_idx, dist in enumerate(distances):
                dist_data = hour_data[hour_data['x'] == dist]['Wm2Front'].values
                
                if len(dist_data) > 0:
                    if aggregation == 'mean':
                        data_by_hour[hour_idx, dist_idx] = np.nanmean(dist_data)
                    elif aggregation == 'max':
                        data_by_hour[hour_idx, dist_idx] = np.nanmax(dist_data)
                    elif aggregation == 'min':
                        data_by_hour[hour_idx, dist_idx] = np.nanmin(dist_data)
                    elif aggregation == 'sum':
                        data_by_hour[hour_idx, dist_idx] = np.nansum(dist_data)
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Create heatmap with rows for specified hours
        im = ax.imshow(data_by_hour, aspect='auto', cmap=cmap,
                      extent=[min(distances), max(distances), -0.5, len(hours_to_show) - 0.5],
                      origin='lower', interpolation='nearest')
        
        # Set labels
        ax.set_xlabel('Location in row-to-row pitch (m)', fontsize=10)
        ax.set_ylabel('Hour of Day', fontsize=10)
        ax.set_yticks(range(len(hours_to_show)))
        ax.set_yticklabels([f'{h:02d}' for h in hours_to_show])
        ax.tick_params(axis='both', labelsize=8)
        
        data_source_str = f" ({data_source})" if data_source else ""
        
    else:
        # Show all time points on y-axis
        # Create pivot table: time x distance
        pivot_data = filtered_df.pivot_table(
            values='Wm2Front',
            index='datetime',
            columns='x',
            aggfunc='mean'  # If multiple values exist for same time/distance, take mean
        )
        
        # Reindex to ensure all distances are included
        pivot_data = pivot_data.reindex(columns=distances)
        
        # Sort by datetime
        pivot_data = pivot_data.sort_index()
        
        # Convert to numpy array
        data = pivot_data.values
        
        # Create heatmap with time on y-axis and distance on x-axis
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                      extent=[min(distances), max(distances), 
                             0, len(times) - 1],
                      origin='lower', interpolation='nearest')
        
        # Set labels
        ax.set_xlabel('Location in row-to-row pitch (m)', fontsize=10)
        ax.set_ylabel('Time Index', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        
        data_source_str = f" ({data_source})" if data_source else ""
        
        # Format time axis
        try:
            from matplotlib.dates import DateFormatter
            # Set yticks to show some representative times
            n_ticks = min(10, len(times))
            tick_indices = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([times[i].strftime('%Y-%m-%d %H:%M') for i in tick_indices])
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        except:
            pass
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ground irradiance (W/m²)', fontsize=10, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    """Command-line interface for plotting irradiance heatmaps."""
    parser = argparse.ArgumentParser(
        description='Plot irradiance heatmap from all_results.pkl or br_full_resolution_results.pkl',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot heatmap aggregated by hour for setup 1, GID 886847, bifacial_radiance data:
  python plot_irradiance_heatmap.py --setup 11 --gid 886847 --data-source bifacial_radiance --data-file all_results_config_11.pkl --output setup_11_heatmap.png

  # Plot full resolution BR dataset (100 distance points):
  python plot_irradiance_heatmap.py --setup 1 --gid 886847 --data-file br_full_resolution_results.pkl --output br_full_res_heatmap.png

  # Plot full time series heatmap for pysam data:
  python plot_irradiance_heatmap.py --setup 1 --gid 886847 --data-source pysam --no-aggregate-by-hour --output plot.png

  # Plot with custom aggregation method:
  python plot_irradiance_heatmap.py --setup 1 --gid 886847 --aggregation max --output plot.png

  # Plot with specified hour range (9 AM to 3 PM):
  python plot_irradiance_heatmap.py --setup 1 --gid 886847 --hour-start 9 --hour-end 15 --output plot.png
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='all_results.pkl',
        help='Path to pickle file (all_results.pkl or br_full_resolution_results.pkl, default: all_results.pkl)'
    )
    
    parser.add_argument(
        '--setup',
        type=int,
        required=True,
        help='Setup number to plot'
    )
    
    parser.add_argument(
        '--gid',
        type=int,
        required=True,
        help='GID to plot'
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['bifacial_radiance', 'pysam'],
        default=None,
        help='Filter by data_source (bifacial_radiance or pysam). If not specified, uses all data sources.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot. If not specified, displays the plot interactively.'
    )
    
    parser.add_argument(
        '--no-aggregate-by-hour',
        action='store_true',
        help='If set, show all time points instead of aggregating by hour of day'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'max', 'min', 'sum'],
        default='mean',
        help='Aggregation method when aggregating by hour (default: mean)'
    )
    
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[3, 2.3],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 3 2.3)'
    )
    
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap to use for the heatmap (default: viridis)'
    )
    
    parser.add_argument(
        '--hour-start',
        type=int,
        default=None,
        metavar='HOUR',
        help='Start hour of day (0-23) for y-axis range. If not specified, uses hours with data.'
    )
    
    parser.add_argument(
        '--hour-end',
        type=int,
        default=None,
        metavar='HOUR',
        help='End hour of day (0-23, inclusive) for y-axis range. If not specified, uses hours with data.'
    )
    
    args = parser.parse_args()
    
    # Validate hour range arguments
    if (args.hour_start is not None and args.hour_end is None) or \
       (args.hour_start is None and args.hour_end is not None):
        print("Error: --hour-start and --hour-end must be provided together.")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    try:
        df = pd.read_pickle(args.data_file)
    except FileNotFoundError:
        print(f"Error: File {args.data_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {args.data_file}: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows")
    print(f"Found {df['gid'].nunique()} unique GIDs")
    print(f"Found {df['setup'].nunique()} unique setups")
    
    # Show distance point information for the selected setup/gid
    if args.setup in df['setup'].values and args.gid in df['gid'].values:
        filtered = df[(df['setup'] == args.setup) & (df['gid'] == args.gid)]
        if args.data_source:
            filtered = filtered[filtered['data_source'] == args.data_source]
        if len(filtered) > 0:
            num_distances = filtered['x'].nunique()
            print(f"Found {num_distances} unique distance points for setup {args.setup}, GID {args.gid}")
            if num_distances > 50:
                print("  (Full resolution dataset detected)")
    
    # Create plot
    try:
        plot_irradiance_heatmap_from_pkl(
            df=df,
            setup=args.setup,
            gid=args.gid,
            data_source=args.data_source,
            output_file=args.output,
            aggregate_by_hour=not args.no_aggregate_by_hour,
            aggregation=args.aggregation,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
            hour_start=args.hour_start,
            hour_end=args.hour_end
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

