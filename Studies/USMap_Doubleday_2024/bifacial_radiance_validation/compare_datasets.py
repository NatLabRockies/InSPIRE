"""
Compare Validation Results and PySAM Outputs
This script compares two solar irradiance datasets from a combined pickle file:
- Bifacial Radiance validation data
- PySAM model outputs from S3 zarr

The combined pickle file should have a 'data_source' column indicating 'bifacial_radiance' or 'pysam'.
Timestamps are already aligned in the consolidated data (bifacial_radiance timestamps are on the hour,
and pysam timestamps are in year 2023).

The script:
- Matches data by gid, setup, and x (distance) coordinates
- Filters to only sun-up times (times present in bifacial_radiance data)
- Calculates summary statistics: MBD, RMSE, MAD (all in both percentage and absolute)

Usage:
    python compare_datasets.py
    python compare_datasets.py --data-file all_results.pkl
    python compare_datasets.py --output comparison_results.csv
    python compare_datasets.py --plot-only --hour-csv comparison_results_by_hour_overall.csv --setup-csv comparison_results_by_hour_setup.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings
import matplotlib.pyplot as plt


def MBD(meas, model):
    """
    Mean Bias Difference (percentage)
    MBD = 100 * [((1/m) * sum(y_i - x_i)) / mean(x_i)]
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Bias Difference as percentage
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate MBD using vectorized operations
    out = 100 * ((1/m) * np.sum(model - meas)) / np.mean(meas)
    
    return out


def RMSE(meas, model):
    """
    Root Mean Squared Error (percentage)
    RMSE = 100 * sqrt((1/m) * sum((y_i - x_i)^2)) / mean(x_i)
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Root Mean Squared Error as percentage
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate RMSE using vectorized operations
    out = 100 * np.sqrt((1/m) * np.sum((model - meas)**2)) / np.mean(meas)
    
    return out


def MBD_abs(meas, model):
    """
    Mean Bias Difference (absolute, not percentage)
    MBD = (1/m) * sum(y_i - x_i)
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Bias Difference (absolute)
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate MBD (absolute) using vectorized operations
    out = (1/m) * np.sum(model - meas)
    
    return out


def RMSE_abs(meas, model):
    """
    Root Mean Squared Error (absolute, not percentage)
    RMSE = sqrt((1/m) * sum((y_i - x_i)^2))
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Root Mean Squared Error (absolute)
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate RMSE (absolute) using vectorized operations
    out = np.sqrt((1/m) * np.sum((model - meas)**2))
    
    return out


def MAD(meas, model):
    """
    Mean Absolute Difference (percentage)
    MAD = 100 * [((1/m) * sum(|y_i - x_i|)) / mean(x_i)]
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Absolute Difference as percentage
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate MAD using vectorized operations
    out = 100 * ((1/m) * np.sum(np.abs(model - meas))) / np.mean(meas)
    
    return out


def MAD_abs(meas, model):
    """
    Mean Absolute Difference (absolute, not percentage)
    MAD = (1/m) * sum(|y_i - x_i|)
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Absolute Difference (absolute)
    """
    # Convert to numpy arrays for efficiency
    meas = np.asarray(meas, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    
    # Remove NaN values using vectorized mask
    mask = ~(np.isnan(meas) | np.isnan(model))
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = np.min(meas)
    mask = model > minirr
    meas = meas[mask]
    model = model[mask]
    
    if len(meas) == 0:
        return np.nan
    
    m = len(meas)
    
    # Calculate MAD (absolute) using vectorized operations
    out = (1/m) * np.sum(np.abs(model - meas))
    
    return out


def plot_metrics_vs_hour(hour_summary_df, output_file=None):
    """
    Plot MBD_absolute and MAD_absolute vs hour of day.
    
    Creates a combined plot with two subplots side-by-side showing both metrics.
    Uses consistent aesthetics with plot_wm2front_vs_distance.py.
    
    Parameters
    ----------
    hour_summary_df : pd.DataFrame
        DataFrame with hour-based statistics including 'hour', 'MBD_absolute', and 'MAD_absolute' columns
    output_file : str, optional
        Base output file path (plot will be saved with _metrics_vs_hour.png suffix)
    
    Returns
    -------
    str or None
        Path to saved plot file, or None if plotting failed
    """
    if len(hour_summary_df) == 0:
        warnings.warn("No data available for plotting metrics vs hour")
        return None
    
    # Ensure hour column exists and data is sorted
    if 'hour' not in hour_summary_df.columns:
        warnings.warn("'hour' column not found in hour_summary_df")
        return None
    
    hour_summary_df = hour_summary_df.sort_values('hour').copy()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Error Metrics vs Hour of Day (Overall)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: MBD_absolute vs hour
    if 'MBD_absolute' in hour_summary_df.columns:
        ax1.plot(hour_summary_df['hour'], hour_summary_df['MBD_absolute'], 
                'o-', linewidth=2, markersize=6, 
                color="#0079C2", alpha=0.7, label='MBD')
        ax1.set_xlabel('Hour of Day', fontsize=10)
        ax1.set_ylabel('Mean Bias Difference (W/m²)', fontsize=10)
        ax1.set_title('MBD (Absolute) vs Hour of Day', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        
        # Add zero line
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax1.text(0.5, 0.5, 'MBD_absolute data not available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: MAD_absolute vs hour
    if 'MAD_absolute' in hour_summary_df.columns:
        ax2.plot(hour_summary_df['hour'], hour_summary_df['MAD_absolute'], 
                's-', linewidth=2, markersize=6, 
                color="#F7A11A", alpha=0.7, label='MAD')
        ax2.set_xlabel('Hour of Day', fontsize=10)
        ax2.set_ylabel('Mean Absolute Difference (W/m²)', fontsize=10)
        ax2.set_title('MAD (Absolute) vs Hour of Day', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        ax2.set_xlim(-0.5, 23.5)
        ax2.set_xticks(range(0, 24, 2))
    else:
        ax2.text(0.5, 0.5, 'MAD_absolute data not available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save combined plot
    if output_file is None:
        plot_path = 'metrics_vs_hour.png'
    else:
        # Handle both .csv and .png extensions
        if output_file.endswith('.png'):
            plot_path = output_file
        elif output_file.endswith('.csv'):
            base_path = output_file.replace('.csv', '')
            plot_path = f'{base_path}_metrics_vs_hour.png'
        else:
            # No extension or other extension - add .png
            plot_path = f'{output_file}_metrics_vs_hour.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Metrics vs hour plot saved to: {plot_path}")
    
    plt.close()
    
    return plot_path


def plot_metrics_vs_hour_by_setup(hour_summary_by_setup_df, output_file=None):
    """
    Plot MBD_absolute and MAD_absolute vs hour of day, tiled by setup.
    
    Creates a 2x5 grid of subplots (one for each setup 1-10) showing both metrics.
    Uses consistent aesthetics with plot_wm2front_vs_distance.py.
    
    Parameters
    ----------
    hour_summary_by_setup_df : pd.DataFrame
        DataFrame with hour-based statistics by setup including 'setup', 'hour', 
        'MBD_absolute', and 'MAD_absolute' columns
    output_file : str, optional
        Base output file path (plot will be saved with _metrics_vs_hour_by_setup.png suffix)
    
    Returns
    -------
    str or None
        Path to saved plot file, or None if plotting failed
    """
    if len(hour_summary_by_setup_df) == 0:
        warnings.warn("No data available for plotting metrics vs hour by setup")
        return None
    
    # Ensure required columns exist
    required_cols = ['setup', 'hour', 'MBD_absolute', 'MAD_absolute']
    missing_cols = [col for col in required_cols if col not in hour_summary_by_setup_df.columns]
    if missing_cols:
        warnings.warn(f"Missing required columns: {missing_cols}")
        return None
    
    # Create figure with subplots for each setup (2 rows, 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('Error Metrics vs Hour of Day by Setup', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Process each setup (1-10)
    for setup in range(1, 11):
        ax = axes[setup - 1]
        
        # Filter data for this setup
        setup_data = hour_summary_by_setup_df[
            hour_summary_by_setup_df['setup'] == setup
        ].sort_values('hour')
        
        if len(setup_data) == 0:
            ax.text(0.5, 0.5, f'Setup {setup}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Setup {setup}', fontweight='bold', fontsize=12)
            continue
        
        # Plot MBD_absolute vs hour
        if 'MBD_absolute' in setup_data.columns:
            ax.plot(setup_data['hour'], setup_data['MBD_absolute'], 
                   'o-', linewidth=2, markersize=6, 
                   color="#0079C2", alpha=0.7, label='MBD')
        
        # Plot MAD_absolute vs hour
        if 'MAD_absolute' in setup_data.columns:
            ax.plot(setup_data['hour'], setup_data['MAD_absolute'], 
                   's-', linewidth=2, markersize=6, 
                   color="#F7A11A", alpha=0.7, label='MAD')
        
        # Formatting
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Error Metric (W/m²)', fontsize=10)
        ax.set_title(f'Setup {setup}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))
        
        # Add zero line for reference
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set reasonable y-axis limits
        if len(setup_data) > 0:
            all_y = []
            if 'MBD_absolute' in setup_data.columns:
                all_y.extend(setup_data['MBD_absolute'].dropna().values)
            if 'MAD_absolute' in setup_data.columns:
                all_y.extend(setup_data['MAD_absolute'].dropna().values)
            
            if len(all_y) > 0:
                y_min, y_max = min(all_y), max(all_y)
                y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
                # Add some padding
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        plot_path = 'metrics_vs_hour_by_setup.png'
    else:
        # Handle both .csv and .png extensions
        if output_file.endswith('.png'):
            base_path = output_file.replace('.png', '')
            plot_path = f'{base_path}_metrics_vs_hour_by_setup.png'
        elif output_file.endswith('.csv'):
            base_path = output_file.replace('.csv', '')
            plot_path = f'{base_path}_metrics_vs_hour_by_setup.png'
        else:
            # No extension or other extension - add .png
            plot_path = f'{output_file}_metrics_vs_hour_by_setup.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Metrics vs hour by setup plot saved to: {plot_path}")
    
    plt.close()
    
    return plot_path




def match_x_values(validation_x, pysam_x, validation_irr, pysam_irr):
    """
    Match x (distance) values between validation and pysam datasets.
    Validation has irregular x values, pysam has regular integer x values.
    Uses interpolation to match validation x values to nearest pysam x values.
    
    Parameters
    ----------
    validation_x : array-like
        Validation x coordinates
    pysam_x : array-like
        PySAM x coordinates (sorted)
    validation_irr : array-like
        Validation irradiance values
    pysam_irr : array-like
        PySAM irradiance values
    
    Returns
    -------
    tuple
        (matched_validation_irr, matched_pysam_irr) arrays
    """
    # Convert to numpy arrays
    val_x = np.array(validation_x)
    pysam_x_sorted = np.sort(np.unique(pysam_x))
    val_irr = np.array(validation_irr)
    pysam_irr = np.array(pysam_irr)
    
    # Create mapping: for each validation x, find closest pysam x
    matched_val_irr = []
    matched_pysam_irr = []
    
    for vx, virr in zip(val_x, val_irr):
        # Find closest pysam x value
        closest_idx = np.argmin(np.abs(pysam_x_sorted - vx))
        closest_pysam_x = pysam_x_sorted[closest_idx]
        
        # Find pysam irradiance at this x value
        pysam_mask = (pysam_x == closest_pysam_x)
        if pysam_mask.sum() > 0:
            # If multiple matches, take the first (should be same value)
            pysam_irr_val = pysam_irr[pysam_mask][0]
            matched_val_irr.append(virr)
            matched_pysam_irr.append(pysam_irr_val)
    
    return np.array(matched_val_irr), np.array(matched_pysam_irr)


def compare_datasets(data_file='all_results.pkl', output_file=None):
    """
    Compare bifacial radiance and pysam outputs from a combined pickle file.
    
    Parameters
    ----------
    data_file : str
        Path to combined results pickle file with data_source column
    output_file : str, optional
        Path to output CSV file for summary statistics
    
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    print("Loading dataset...")
    
    # Load combined dataset
    all_data = pd.read_pickle(data_file)
    
    # Convert datetime columns
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    
    # Split into bifacial_radiance and pysam datasets
    # Use views initially, only copy when modifying
    validation = all_data[all_data['data_source'] == 'bifacial_radiance']
    pysam = all_data[all_data['data_source'] == 'pysam']
    
    print(f"Bifacial Radiance data: {len(validation)} rows, {validation['datetime'].nunique()} unique times")
    print(f"PySAM data: {len(pysam)} rows, {pysam['datetime'].nunique()} unique times")
    
    # Extract day of year and hour (ignoring year)
    validation['dayofyear'] = validation['datetime'].dt.dayofyear
    validation['hour'] = validation['datetime'].dt.hour
    validation['minute'] = validation['datetime'].dt.minute
    
    pysam['dayofyear'] = pysam['datetime'].dt.dayofyear
    pysam['hour'] = pysam['datetime'].dt.hour
    
    # Get unique validation times (these are the "sun-up" times)
    validation_times = validation['datetime'].unique()
    print(f"\nValidation time range: {validation_times.min()} to {validation_times.max()}")
    print(f"Validation day of year range: {validation['dayofyear'].min()} to {validation['dayofyear'].max()}")
    print(f"Validation hours: {sorted(validation['hour'].unique())}")
    
    # Get unique combinations of gid, setup
    gid_setup_combos = validation[['gid', 'setup']].drop_duplicates()
    print(f"\nFound {len(gid_setup_combos)} unique (gid, setup) combinations")
    
    # Initialize results list
    results = []
    all_matched_data = []  # Store all matched data for reuse
    
    # Process each (gid, setup) combination using itertuples (much faster than iterrows)
    for row in gid_setup_combos.itertuples(index=False):
        gid = row.gid
        setup = row.setup
        
        print(f"\nProcessing GID {gid}, Setup {setup}...")
        
        # Filter data for this gid and setup (no need to copy if not modifying)
        val_subset = validation[
            (validation['gid'] == gid) & 
            (validation['setup'] == setup)
        ]
        
        pysam_subset = pysam[
            (pysam['gid'] == gid) & 
            (pysam['setup'] == setup)
        ]
        
        if len(val_subset) == 0 or len(pysam_subset) == 0:
            warnings.warn(f"No data for GID {gid}, Setup {setup}")
            continue
        
        # Match by day of year, hour, and x
        matched_data = []
        
        # Get unique combinations of day of year and hour
        for (doy, hour), val_group in val_subset.groupby(['dayofyear', 'hour']):
            # Find corresponding pysam data (same day of year and hour)
            pysam_time_data = pysam_subset[
                (pysam_subset['dayofyear'] == doy) & 
                (pysam_subset['hour'] == hour)
            ]
            
            if len(pysam_time_data) == 0:
                continue
            
            # Vectorized x-value matching using searchsorted
            # Sort pysam data by x once per time group
            pysam_sorted = pysam_time_data.sort_values('x')
            pysam_x_sorted = pysam_sorted['x'].values
            pysam_irr_sorted = pysam_sorted['Wm2Front'].values
            
            # Get validation x and irradiance values as arrays
            val_x = val_group['x'].values
            val_irr = val_group['Wm2Front'].values
            
            # Use searchsorted to find insertion points (nearest neighbor)
            indices = np.searchsorted(pysam_x_sorted, val_x, side='left')
            
            # Handle edge cases: clamp indices to valid range
            indices = np.clip(indices, 0, len(pysam_x_sorted) - 1)
            
            # For each validation point, check if left or right neighbor is closer
            left_dist = np.abs(pysam_x_sorted[np.maximum(indices - 1, 0)] - val_x)
            right_dist = np.abs(pysam_x_sorted[indices] - val_x)
            
            # Choose closer neighbor
            use_left = (left_dist < right_dist) & (indices > 0)
            final_indices = np.where(use_left, indices - 1, indices)
            
            # Get matched pysam irradiance values
            matched_pysam_irr = pysam_irr_sorted[final_indices]
            
            # Create matched data entries using vectorized operations
            n_matches = len(val_x)
            matched_data.extend([
                {
                    'gid': gid,
                    'setup': setup,
                    'dayofyear': doy,
                    'hour': hour,
                    'x': val_x[i],
                    'validation_irr': val_irr[i],
                    'pysam_irr': matched_pysam_irr[i]
                }
                for i in range(n_matches)
            ])
        
        if len(matched_data) == 0:
            warnings.warn(f"No matched data for GID {gid}, Setup {setup}")
            continue
        
        matched_df = pd.DataFrame(matched_data)
        all_matched_data.extend(matched_data)  # Store for reuse
        
        # Calculate statistics for this gid/setup combination
        meas = matched_df['validation_irr'].values
        model = matched_df['pysam_irr'].values
        
        # Filter out zero or very small values for meaningful comparison
        mask = (meas > 0.1) & (model > 0.1)
        meas_filtered = meas[mask]
        model_filtered = model[mask]
        
        if len(meas_filtered) == 0:
            warnings.warn(f"No valid data points for GID {gid}, Setup {setup} after filtering")
            continue
        
        # Calculate statistics
        mbd_pct = MBD(meas_filtered, model_filtered)
        rmse_pct = RMSE(meas_filtered, model_filtered)
        mad_pct = MAD(meas_filtered, model_filtered)
        mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
        rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
        mad_abs_val = MAD_abs(meas_filtered, model_filtered)
        
        # Additional statistics
        n_points = len(meas_filtered)
        mean_meas = meas_filtered.mean()
        mean_model = model_filtered.mean()
        corr = np.corrcoef(meas_filtered, model_filtered)[0, 1]
        
        results.append({
            'gid': gid,
            'setup': setup,
            'n_points': n_points,
            'mean_validation': mean_meas,
            'mean_pysam': mean_model,
            'MBD_percent': mbd_pct,
            'RMSE_percent': rmse_pct,
            'MAD_percent': mad_pct,
            'MBD_absolute': mbd_abs_val,
            'RMSE_absolute': rmse_abs_val,
            'MAD_absolute': mad_abs_val,
            'correlation': corr
        })
        
        print(f"  Matched {n_points} data points")
        print(f"  MBD: {mbd_pct:.2f}%, RMSE: {rmse_pct:.2f}%, MAD: {mad_pct:.2f}%")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    if len(summary_df) == 0:
        raise ValueError("No matched data found. Check data alignment.")
    
    # Add overall statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (by GID and Setup)")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Calculate statistics by GID only (across all setups, distances, and hours)
    print("\n" + "="*60)
    print("Calculating statistics by GID (across all setups, distances, and hours)...")
    print("="*60)
    
    gid_results = []
    gid_summary_df = pd.DataFrame()  # Initialize as empty DataFrame
    
    if len(all_matched_data) > 0:
        all_matched_df = pd.DataFrame(all_matched_data)
        
        # Group by GID only
        for gid, group in all_matched_df.groupby('gid'):
            meas = group['validation_irr'].values
            model = group['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mad_pct = MAD(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            mad_abs_val = MAD_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            gid_results.append({
                'gid': gid,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MAD_percent': mad_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'MAD_absolute': mad_abs_val,
                'correlation': corr
            })
    
    gid_summary_df = pd.DataFrame(gid_results)
    
    if len(gid_summary_df) > 0:
        # Sort by gid
        gid_summary_df = gid_summary_df.sort_values('gid').reset_index(drop=True)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS (by GID)")
        print("="*60)
        print(gid_summary_df.to_string(index=False))
    else:
        print("No GID-based statistics calculated.")
    
    # Calculate matched time-points per GID
    print("\n" + "="*60)
    print("MATCHED TIME-POINTS PER GID")
    print("="*60)
    
    if len(all_matched_data) > 0:
        all_matched_df = pd.DataFrame(all_matched_data)
        
        # Count unique time-points (dayofyear + hour) per GID using vectorized operations
        timepoint_counts = []
        for gid in all_matched_df['gid'].unique():
            gid_mask = all_matched_df['gid'] == gid
            gid_data = all_matched_df[gid_mask]
            
            # Count unique combinations of dayofyear and hour using vectorized operations
            unique_timepoints = gid_data[['dayofyear', 'hour']].drop_duplicates()
            n_timepoints = len(unique_timepoints)
            
            # Also get the range of times using vectorized operations
            unique_days = np.sort(gid_data['dayofyear'].unique())
            unique_hours = np.sort(gid_data['hour'].unique())
            
            timepoint_counts.append({
                'gid': gid,
                'n_matched_timepoints': n_timepoints,
                'dayofyear_range': f"{unique_days[0]}-{unique_days[-1]}" if len(unique_days) > 1 else str(unique_days[0]),
                'hour_range': f"{unique_hours[0]}-{unique_hours[-1]}" if len(unique_hours) > 1 else str(unique_hours[0]),
                'unique_days': len(unique_days),
                'unique_hours': len(unique_hours)
            })
        
        timepoint_summary = pd.DataFrame(timepoint_counts).sort_values('gid')
        print(timepoint_summary.to_string(index=False))
        
        # Save to file
        if output_file:
            timepoint_output_file = output_file.replace('.csv', '_timepoints_per_gid.csv')
            timepoint_summary.to_csv(timepoint_output_file, index=False)
            print(f"\nTime-point summary saved to: {timepoint_output_file}")
    else:
        print("No matched data available for time-point analysis.")
    
    # Calculate statistics by hour of day
    print("\n" + "="*60)
    print("Calculating statistics by hour of day...")
    print("="*60)
    
    hour_results_overall = []
    hour_results_by_setup = []
    
    if len(all_matched_data) > 0:
        all_matched_df = pd.DataFrame(all_matched_data)
        
        # Overall statistics by hour (averaging across all GIDs, setups, distances)
        for hour in sorted(all_matched_df['hour'].unique()):
            hour_data = all_matched_df[all_matched_df['hour'] == hour]
            
            meas = hour_data['validation_irr'].values
            model = hour_data['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mad_pct = MAD(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            mad_abs_val = MAD_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            hour_results_overall.append({
                'hour': hour,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MAD_percent': mad_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'MAD_absolute': mad_abs_val,
                'correlation': corr
            })
        
        # Statistics by hour and setup (averaging over GID and distance)
        for (setup, hour), group in all_matched_df.groupby(['setup', 'hour']):
            meas = group['validation_irr'].values
            model = group['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mad_pct = MAD(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            mad_abs_val = MAD_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            hour_results_by_setup.append({
                'setup': setup,
                'hour': hour,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MAD_percent': mad_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'MAD_absolute': mad_abs_val,
                'correlation': corr
            })
        
        # Create DataFrames
        hour_summary_overall = pd.DataFrame(hour_results_overall).sort_values('hour')
        hour_summary_by_setup = pd.DataFrame(hour_results_by_setup).sort_values(['setup', 'hour'])
        
        # Print overall statistics by hour
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY HOUR (Overall - All GIDs, Setups, Distances)")
        print("="*60)
        print(hour_summary_overall.to_string(index=False))
        
        # Print statistics by hour and setup
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY HOUR AND SETUP (Averaged over GID and Distance)")
        print("="*60)
        print(hour_summary_by_setup.to_string(index=False))
        
        # Save to files
        if output_file:
            hour_overall_output = output_file.replace('.csv', '_by_hour_overall.csv')
            hour_summary_overall.to_csv(hour_overall_output, index=False)
            print(f"\nHour statistics (overall) saved to: {hour_overall_output}")
            
            hour_setup_output = output_file.replace('.csv', '_by_hour_setup.csv')
            hour_summary_by_setup.to_csv(hour_setup_output, index=False)
            print(f"Hour statistics (by hour and setup) saved to: {hour_setup_output}")
        
        # Generate plots for metrics vs hour
        print("\n" + "="*60)
        print("Generating plots: Metrics vs Hour of Day")
        print("="*60)
        plot_metrics_vs_hour(hour_summary_overall, output_file=output_file)
        
        # Generate plots for metrics vs hour by setup
        print("\n" + "="*60)
        print("Generating plots: Metrics vs Hour of Day by Setup")
        print("="*60)
        plot_metrics_vs_hour_by_setup(hour_summary_by_setup, output_file=output_file)
    else:
        print("No matched data available for hour-based analysis.")
    
    # Calculate overall statistics across all gid/setup combinations
    # Reuse matched data instead of re-matching
    if len(all_matched_data) > 0:
        all_matched_df = pd.DataFrame(all_matched_data)
        
        # Filter out zero or very small values
        mask = (all_matched_df['validation_irr'] > 0.1) & (all_matched_df['pysam_irr'] > 0.1)
        all_matched_filtered = all_matched_df[mask]
        
        if len(all_matched_filtered) > 0:
            all_meas = all_matched_filtered['validation_irr'].values
            all_model = all_matched_filtered['pysam_irr'].values
        else:
            all_meas = np.array([])
            all_model = np.array([])
    else:
        all_meas = np.array([])
        all_model = np.array([])
    
    if len(all_meas) > 0:
        all_meas = np.array(all_meas)
        all_model = np.array(all_model)
        
        overall_mbd_pct = MBD(all_meas, all_model)
        overall_rmse_pct = RMSE(all_meas, all_model)
        overall_mad_pct = MAD(all_meas, all_model)
        overall_mbd_abs = MBD_abs(all_meas, all_model)
        overall_rmse_abs = RMSE_abs(all_meas, all_model)
        overall_mad_abs = MAD_abs(all_meas, all_model)
        overall_corr = np.corrcoef(all_meas, all_model)[0, 1]
        
        overall_row = {
            'gid': 'ALL',
            'setup': 'ALL',
            'n_points': len(all_meas),
            'mean_validation': all_meas.mean(),
            'mean_pysam': all_model.mean(),
            'MBD_percent': overall_mbd_pct,
            'RMSE_percent': overall_rmse_pct,
            'MAD_percent': overall_mad_pct,
            'MBD_absolute': overall_mbd_abs,
            'RMSE_absolute': overall_rmse_abs,
            'MAD_absolute': overall_mad_abs,
            'correlation': overall_corr
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
        
        print("\n" + "="*60)
        print("OVERALL STATISTICS (All GIDs and Setups Combined)")
        print("="*60)
        print(f"Total matched points: {len(all_meas)}")
        print(f"MBD: {overall_mbd_pct:.2f}%")
        print(f"RMSE: {overall_rmse_pct:.2f}%")
        print(f"MAD: {overall_mad_pct:.2f}%")
        print(f"MBD (absolute): {overall_mbd_abs:.2f} W/m²")
        print(f"RMSE (absolute): {overall_rmse_abs:.2f} W/m²")
        print(f"MAD (absolute): {overall_mad_abs:.2f} W/m²")
        print(f"Correlation: {overall_corr:.4f}")
    
    # Save to file if requested
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary statistics (by GID and Setup) saved to: {output_file}")
        
        # Save GID-based statistics to a separate file
        if len(gid_summary_df) > 0:
            gid_output_file = output_file.replace('.csv', '_by_gid.csv')
            gid_summary_df.to_csv(gid_output_file, index=False)
            print(f"Summary statistics (by GID) saved to: {gid_output_file}")
    
    return summary_df, gid_summary_df


def plot_from_csv(hour_overall_csv=None, hour_setup_csv=None, output_file=None):
    """
    Load hour-based statistics from CSV files and generate plots.
    
    Parameters
    ----------
    hour_overall_csv : str, optional
        Path to CSV file with overall hour-based statistics (e.g., comparison_results_by_hour_overall.csv)
    hour_setup_csv : str, optional
        Path to CSV file with hour-based statistics by setup (e.g., comparison_results_by_hour_setup.csv)
    output_file : str, optional
        Base output file path for plots (default: based on CSV filenames)
    
    Returns
    -------
    tuple
        Tuple of (overall_plot_path, setup_plot_path), either may be None
    """
    overall_plot_path = None
    setup_plot_path = None
    
    # Plot overall metrics vs hour
    if hour_overall_csv:
        print(f"Loading overall hour statistics from: {hour_overall_csv}")
        
        try:
            hour_summary_df = pd.read_csv(hour_overall_csv)
        except Exception as e:
            warnings.warn(f"Failed to load CSV file {hour_overall_csv}: {e}")
        else:
            if len(hour_summary_df) == 0:
                warnings.warn(f"CSV file {hour_overall_csv} is empty")
            else:
                # Determine output file path
                if output_file is None:
                    # Generate output filename based on input CSV
                    csv_path = Path(hour_overall_csv)
                    plot_output = csv_path.parent / f"{csv_path.stem}_metrics_vs_hour.png"
                    plot_output = str(plot_output)
                else:
                    plot_output = output_file
                
                print(f"Generating plot: Metrics vs Hour of Day (Overall)")
                overall_plot_path = plot_metrics_vs_hour(hour_summary_df, output_file=plot_output)
    
    # Plot metrics vs hour by setup
    if hour_setup_csv:
        print(f"Loading hour statistics by setup from: {hour_setup_csv}")
        
        try:
            hour_setup_df = pd.read_csv(hour_setup_csv)
        except Exception as e:
            warnings.warn(f"Failed to load CSV file {hour_setup_csv}: {e}")
        else:
            if len(hour_setup_df) == 0:
                warnings.warn(f"CSV file {hour_setup_csv} is empty")
            else:
                # Determine output file path
                if output_file is None:
                    # Generate output filename based on input CSV
                    csv_path = Path(hour_setup_csv)
                    plot_output = csv_path.parent / f"{csv_path.stem}_metrics_vs_hour_by_setup.png"
                    plot_output = str(plot_output)
                else:
                    plot_output = output_file
                
                print(f"Generating plot: Metrics vs Hour of Day by Setup")
                setup_plot_path = plot_metrics_vs_hour_by_setup(hour_setup_df, output_file=plot_output)
    
    return overall_plot_path, setup_plot_path


def main():
    """Command-line interface for comparing datasets."""
    parser = argparse.ArgumentParser(
        description='Compare validation results and PySAM outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison with plotting
  python compare_datasets.py
  python compare_datasets.py --data-file all_results.pkl
  python compare_datasets.py --output comparison_results.csv
  
  # Plot only from pre-calculated CSV files
  # Generate overall plot only
  python compare_datasets.py --plot-only --hour-csv comparison_results_by_hour_overall.csv
  
  # Generate by-setup plot only
  python compare_datasets.py --plot-only --setup-csv comparison_results_by_hour_setup.csv
  
  # Generate both plots (recommended)
  python compare_datasets.py --plot-only --hour-csv comparison_results_by_hour_overall.csv --setup-csv comparison_results_by_hour_setup.csv
  
  # Generate both plots with custom output path
  python compare_datasets.py --plot-only --hour-csv comparison_results_by_hour_overall.csv --setup-csv comparison_results_by_hour_setup.csv --output my_plots
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='all_results.pkl',
        help='Path to combined results pickle file with data_source column (default: all_results.pkl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for summary statistics (default: comparison_summary.csv). For plot-only mode, this is the plot output path.'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots from pre-calculated CSV file (requires --hour-csv)'
    )
    
    parser.add_argument(
        '--hour-csv',
        type=str,
        default=None,
        help='Path to pre-calculated overall hour statistics CSV file (e.g., comparison_results_by_hour_overall.csv). Used for overall plot in --plot-only mode.'
    )
    
    parser.add_argument(
        '--setup-csv',
        type=str,
        default=None,
        help='Path to pre-calculated hour statistics by setup CSV file (e.g., comparison_results_by_hour_setup.csv). Used for by-setup plot in --plot-only mode.'
    )
    
    args = parser.parse_args()
    
    # Handle plot-only mode
    if args.plot_only:
        if args.hour_csv is None and args.setup_csv is None:
            parser.error("--plot-only requires at least one of --hour-csv or --setup-csv")
        
        overall_path, setup_path = plot_from_csv(
            hour_overall_csv=args.hour_csv,
            hour_setup_csv=args.setup_csv,
            output_file=args.output
        )
        
        print("\n" + "="*60)
        print("Plot generation complete!")
        print("="*60)
        if overall_path:
            print(f"Overall plot saved to: {overall_path}")
        if setup_path:
            print(f"By-setup plot saved to: {setup_path}")
        
        return overall_path, setup_path
    
    # Normal mode: full comparison
    # Set default output filename if not provided
    output_file = args.output if args.output else 'comparison_summary.csv'
    
    # Compare datasets
    summary, distance_summary = compare_datasets(
        data_file=args.data_file,
        output_file=output_file
    )
    
    return summary, distance_summary


if __name__ == "__main__":
    main()

