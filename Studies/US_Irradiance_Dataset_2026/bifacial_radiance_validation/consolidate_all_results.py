"""
Consolidate All Results Script
This script consolidates both bifacial radiance validation data and S3 zarr data
into a single CSV file with a data_source column indicating the source.

First, it processes bifacial radiance results from the validation_results folder.
Then, it uses the same GIDs to access corresponding results from S3 zarr files.
The zarr distance indices (0-9) are mapped to actual distance values in meters
from the bifacial radiance data.

This script remaps legacy validation setup numbering to the current S3 ordering:
1->1, 2->2, 3->3, 4->4, 5->5, 11->6, 6->7, 7->8, 8->9, 9->10, 10->11.

Usage as Python function:
    from consolidate_all_results import consolidate_all_results
    data = consolidate_all_results("validation_results", 
                                   base_path="bifacial_radiance_validation")

Usage from command line:
    python consolidate_all_results.py validation_results
    python consolidate_all_results.py validation_results --base-path "/path/to/folder"
    python consolidate_all_results.py validation_results --output all_results.pkl
    python consolidate_all_results.py validation_results --br-only-full-res
"""

import pandas as pd
import xarray as xr
import fsspec
from pathlib import Path
from datetime import datetime, timedelta
import re
import warnings
import argparse
import numpy as np


# Setup IDs that use BR `y` as the distance axis in legacy validation numbering.
DISTANCE_ON_Y_SETUPS_VALIDATION = {6, 7, 8, 9, 11}

# Setup IDs that use BR `y` as the distance axis in current S3 numbering.
DISTANCE_ON_Y_SETUPS_S3 = {6, 7, 8, 9, 10}

# Legacy validation setup IDs mapped to current S3 configuration numbering.
# This mapping keeps downstream analysis keyed on S3 setup semantics.
VALIDATION_TO_S3_SETUP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    11: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
}


def remap_validation_setups_to_s3(br_data):
    """
    Remap legacy validation setup IDs to current S3 setup numbering.

    Parameters
    ----------
    br_data : pd.DataFrame
        Consolidated bifacial radiance data with a `setup` column.

    Returns
    -------
    pd.DataFrame
        Copy of input data with `setup` remapped to S3 numbering.
    """
    if 'setup' not in br_data.columns:
        raise ValueError("Expected 'setup' column in bifacial radiance data")

    observed_setups = sorted(pd.unique(br_data['setup']))
    unmapped_setups = [setup for setup in observed_setups if setup not in VALIDATION_TO_S3_SETUP]
    if unmapped_setups:
        raise ValueError(
            "Found setup IDs without validation->S3 mapping: "
            f"{unmapped_setups}. Update VALIDATION_TO_S3_SETUP before continuing."
        )

    remapped_data = br_data.copy()
    remapped_data['setup'] = remapped_data['setup'].map(VALIDATION_TO_S3_SETUP).astype(int)

    print("\nApplying validation->S3 setup remap:")
    for setup in observed_setups:
        print(f"  Setup {setup} -> {VALIDATION_TO_S3_SETUP[setup]}")

    return remapped_data


def _br_distance_axis_for_setup(setup_num: int) -> str:
    """
    Which bifacial_radiance coordinate corresponds to PySAM's `distance` dimension.

    Interprets `setup_num` using legacy validation numbering.
    - Setups 1-5 and 10: BR `x` maps to PySAM `distance`
    - Setups 6-9 and 11: BR `y` maps to PySAM `distance`
    """
    return "y" if setup_num in DISTANCE_ON_Y_SETUPS_VALIDATION else "x"


def consolidate_br_results(folder_name, base_path="."):
    """
    Consolidate bifacial radiance results from validation_results folder.
    
    Parameters
    ----------
    folder_name : str
        Name of the folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the folder. Defaults to current directory.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front
        where `x` is the distance coordinate aligned to PySAM `distance`.
        Contains only timestamps that exist in the CSV files.
    """
    # Construct full path to folder
    folder_path = Path(base_path) / folder_name
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all setup folders (typically numbered 1-11)
    setup_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    setup_numbers = []
    
    for setup_folder in setup_folders:
        try:
            setup_num = int(setup_folder.name)
            setup_numbers.append(setup_num)
        except ValueError:
            continue
    
    setup_numbers = sorted(setup_numbers)
    
    if len(setup_numbers) == 0:
        raise ValueError("No setup folders found")
    
    print(f"Found {len(setup_numbers)} setup folders")
    
    # Initialize list to store data from each setup/GID combination
    all_data = []
    
    # Process each setup
    for setup_num in setup_numbers:
        print(f"Processing setup {setup_num}...")
        
        setup_path = folder_path / str(setup_num)
        
        # Get all GID folders within this setup
        gid_folders = [d for d in setup_path.iterdir() if d.is_dir()]
        
        # Process each GID folder
        for gid_folder in gid_folders:
            gid = gid_folder.name
            print(f"  Processing GID {gid}...")
            
            # Get all day folders within this GID folder
            day_folders = [d for d in gid_folder.iterdir() if d.is_dir()]
            
            if len(day_folders) == 0:
                warnings.warn(f"No day folders found in: {gid_folder}")
                continue
            
            # Process each day folder
            for day_folder in day_folders:
                results_path = day_folder / "results"
                
                if not results_path.exists() or not results_path.is_dir():
                    warnings.warn(f"Results folder not found: {results_path}")
                    continue
                
                # Find all Ground CSV files
                csv_files = list(results_path.glob("*Ground*.csv"))
                
                if len(csv_files) == 0:
                    warnings.warn(f"No Ground CSV files found in: {results_path}")
                    continue
                
                # Process each CSV file
                for csv_file in csv_files:
                    # Extract datetime from filename
                    filename = csv_file.name
                    
                    # Extract date and time from filename
                    datetime_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{4})", filename)
                    
                    if datetime_match is None:
                        warnings.warn(f"Could not extract datetime from: {filename}")
                        continue
                    
                    # Parse datetime (format: YYYY-MM-DD_HHMM)
                    datetime_str = datetime_match.group(1)
                    date_part = re.search(r"\d{4}-\d{2}-\d{2}", datetime_str).group(0)
                    time_part = re.search(r"\d{4}$", datetime_str).group(0)
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    
                    # Create datetime object
                    file_datetime = datetime.strptime(
                        f"{date_part} {hour:02d}:{minute:02d}", 
                        "%Y-%m-%d %H:%M"
                    )
                    
                    # Adjust timestamp: subtract 30 minutes so it occurs on the hour (00:00)
                    file_datetime = file_datetime - timedelta(minutes=30)
                    
                    # Read CSV file
                    try:
                        csv_data = pd.read_csv(
                            csv_file,
                            dtype={
                                'x': float,
                                'y': float,
                                'z': float,
                                'mattype': str,
                                'Wm2Front': float
                            }
                        )

                        # Aggregate to 10 distance points (matching PySAM resolution) per timestamp.
                        # - Setups 1-5 and 10: distance = BR `x`, average Wm2Front across `y` for each `x`, then chunk into 10 segments
                        # - Setups 6-9 and 11: distance = BR `y`, average Wm2Front across `x` for each `y`, then chunk into 10 segments
                        distance_axis = _br_distance_axis_for_setup(setup_num)
                        if distance_axis not in csv_data.columns:
                            raise KeyError(
                                f"Expected column '{distance_axis}' in {csv_file.name}, "
                                f"but found columns: {list(csv_data.columns)}"
                            )

                        # First, average across the non-distance axis (y for setups 1-5,10; x for setups 6-9,11)
                        # This is done implicitly by grouping by distance_axis and averaging Wm2Front
                        pre_aggregated = (
                            csv_data
                            .groupby(distance_axis, as_index=False)["Wm2Front"]
                            .mean()
                            .sort_values(distance_axis)
                            .reset_index(drop=True)
                        )
                        
                        # Chunk the distance axis into 10 segments and average within each chunk
                        num_points = len(pre_aggregated)
                        num_chunks = 10
                        chunk_size = num_points / num_chunks
                        
                        chunked_data = []
                        for chunk_idx in range(num_chunks):
                            start_idx = int(chunk_idx * chunk_size)
                            # For the last chunk, include all remaining points
                            if chunk_idx == num_chunks - 1:
                                end_idx = num_points
                            else:
                                end_idx = int((chunk_idx + 1) * chunk_size)
                            
                            chunk = pre_aggregated.iloc[start_idx:end_idx]
                            
                            # Calculate midpoint of distance values in this chunk
                            distance_midpoint = chunk[distance_axis].mean()
                            
                            # Average Wm2Front across this chunk
                            wm2_avg = chunk["Wm2Front"].mean()
                            
                            chunked_data.append({
                                "x": distance_midpoint,
                                "Wm2Front": wm2_avg
                            })
                        
                        aggregated = pd.DataFrame(chunked_data)
                        
                        # Add metadata columns
                        aggregated['gid'] = int(gid)
                        aggregated['setup'] = setup_num
                        aggregated['datetime'] = file_datetime
                        aggregated['data_source'] = 'bifacial_radiance'
                        
                        # Reorder columns
                        aggregated = aggregated[['gid', 'setup', 'datetime', 'x', 'Wm2Front', 'data_source']]
                        
                        all_data.append(aggregated)
                        
                    except Exception as e:
                        warnings.warn(f"Error reading {csv_file}: {e}")
    
    # Combine all data
    if len(all_data) == 0:
        raise ValueError("No data loaded. Check folder structure and file names.")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nLoaded {len(combined_data)} rows of bifacial radiance data")
    print(f"Found {combined_data['gid'].nunique()} GIDs")
    print(f"Found {combined_data['setup'].nunique()} setups")
    print(f"Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
    
    return combined_data


def consolidate_br_results_full_resolution(folder_name, base_path="."):
    """
    Consolidate bifacial radiance results from validation_results folder at full resolution.
    This version keeps all distance points (typically ~100) instead of aggregating to 10.
    
    Parameters
    ----------
    folder_name : str
        Name of the folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the folder. Defaults to current directory.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front, data_source
        where `x` is the distance coordinate aligned to PySAM `distance`.
        Contains all distance points (full resolution) after averaging across the non-distance axis.
    """
    # Construct full path to folder
    folder_path = Path(base_path) / folder_name
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all setup folders (typically numbered 1-11)
    setup_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    setup_numbers = []
    
    for setup_folder in setup_folders:
        try:
            setup_num = int(setup_folder.name)
            setup_numbers.append(setup_num)
        except ValueError:
            continue
    
    setup_numbers = sorted(setup_numbers)
    
    if len(setup_numbers) == 0:
        raise ValueError("No setup folders found")
    
    print(f"Found {len(setup_numbers)} setup folders")
    
    # Initialize list to store data from each setup/GID combination
    all_data = []
    
    # Process each setup
    for setup_num in setup_numbers:
        print(f"Processing setup {setup_num}...")
        
        setup_path = folder_path / str(setup_num)
        
        # Get all GID folders within this setup
        gid_folders = [d for d in setup_path.iterdir() if d.is_dir()]
        
        # Process each GID folder
        for gid_folder in gid_folders:
            gid = gid_folder.name
            print(f"  Processing GID {gid}...")
            
            # Get all day folders within this GID folder
            day_folders = [d for d in gid_folder.iterdir() if d.is_dir()]
            
            if len(day_folders) == 0:
                warnings.warn(f"No day folders found in: {gid_folder}")
                continue
            
            # Process each day folder
            for day_folder in day_folders:
                results_path = day_folder / "results"
                
                if not results_path.exists() or not results_path.is_dir():
                    warnings.warn(f"Results folder not found: {results_path}")
                    continue
                
                # Find all Ground CSV files
                csv_files = list(results_path.glob("*Ground*.csv"))
                
                if len(csv_files) == 0:
                    warnings.warn(f"No Ground CSV files found in: {results_path}")
                    continue
                
                # Process each CSV file
                for csv_file in csv_files:
                    # Extract datetime from filename
                    filename = csv_file.name
                    
                    # Extract date and time from filename
                    datetime_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{4})", filename)
                    
                    if datetime_match is None:
                        warnings.warn(f"Could not extract datetime from: {filename}")
                        continue
                    
                    # Parse datetime (format: YYYY-MM-DD_HHMM)
                    datetime_str = datetime_match.group(1)
                    date_part = re.search(r"\d{4}-\d{2}-\d{2}", datetime_str).group(0)
                    time_part = re.search(r"\d{4}$", datetime_str).group(0)
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    
                    # Create datetime object
                    file_datetime = datetime.strptime(
                        f"{date_part} {hour:02d}:{minute:02d}", 
                        "%Y-%m-%d %H:%M"
                    )
                    
                    # Adjust timestamp: subtract 30 minutes so it occurs on the hour (00:00)
                    file_datetime = file_datetime - timedelta(minutes=30)
                    
                    # Read CSV file
                    try:
                        csv_data = pd.read_csv(
                            csv_file,
                            dtype={
                                'x': float,
                                'y': float,
                                'z': float,
                                'mattype': str,
                                'Wm2Front': float
                            }
                        )

                        # Keep full resolution: average across the non-distance axis only.
                        # - Setups 1-5 and 10: distance = BR `x`, average Wm2Front across `y` for each `x`
                        # - Setups 6-9 and 11: distance = BR `y`, average Wm2Front across `x` for each `y`
                        distance_axis = _br_distance_axis_for_setup(setup_num)
                        if distance_axis not in csv_data.columns:
                            raise KeyError(
                                f"Expected column '{distance_axis}' in {csv_file.name}, "
                                f"but found columns: {list(csv_data.columns)}"
                            )

                        # Average across the non-distance axis (y for setups 1-5,10; x for setups 6-9,11)
                        # This keeps all distance points at full resolution
                        aggregated = (
                            csv_data
                            .groupby(distance_axis, as_index=False)["Wm2Front"]
                            .mean()
                            .sort_values(distance_axis)
                            .reset_index(drop=True)
                        )
                        
                        # Rename distance_axis column to 'x' for consistency
                        aggregated = aggregated.rename(columns={distance_axis: 'x'})
                        
                        # Add metadata columns
                        aggregated['gid'] = int(gid)
                        aggregated['setup'] = setup_num
                        aggregated['datetime'] = file_datetime
                        aggregated['data_source'] = 'bifacial_radiance'
                        
                        # Reorder columns
                        aggregated = aggregated[['gid', 'setup', 'datetime', 'x', 'Wm2Front', 'data_source']]
                        
                        all_data.append(aggregated)
                        
                    except Exception as e:
                        warnings.warn(f"Error reading {csv_file}: {e}")
    
    # Combine all data
    if len(all_data) == 0:
        raise ValueError("No data loaded. Check folder structure and file names.")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nLoaded {len(combined_data)} rows of bifacial radiance data (full resolution)")
    print(f"Found {combined_data['gid'].nunique()} GIDs")
    print(f"Found {combined_data['setup'].nunique()} setups")
    print(f"Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
    
    return combined_data


def create_distance_mapping(br_data):
    """
    Create a mapping from zarr distance indices (0-9) to actual distance values in meters.
    Uses the distance values from bifacial radiance data, which are monotonically increasing.
    
    Mapping strategy (setup-aware):
    Interprets setup IDs using current S3 numbering.
    - Setups 1-5 and 11: Reverse mapping (index 9 -> smallest distance, index 0 -> largest)
    - Setups 6-10: Direct mapping (index 0 -> smallest distance, index 9 -> largest)
    
    Parameters
    ----------
    br_data : pd.DataFrame
        Bifacial radiance data with 'x' column containing distance values
        aligned to PySAM `distance` (setup-aware: BR x for setups 1-5,10; BR y for setups 6-9,11).
    
    Returns
    -------
    dict
        Dictionary mapping {setup: {gid: {index: distance_value}}}
    """
    print("\nCreating distance index to meter mapping from bifacial radiance data...")
    
    distance_mapping = {}
    
    for setup in sorted(br_data['setup'].unique()):
        setup_data = br_data[br_data['setup'] == setup]
        distance_mapping[setup] = {}
        
        # Determine if reverse mapping is needed using S3 setup numbering.
        # Setups 1-5 and 11 use reverse mapping; setups 6-10 use direct mapping.
        use_reverse = setup not in DISTANCE_ON_Y_SETUPS_S3
        
        for gid in sorted(setup_data['gid'].unique()):
            gid_data = setup_data[setup_data['gid'] == gid]
            
            # Get unique distance values and sort them (monotonically increasing)
            unique_distances = sorted(gid_data['x'].unique())
            
            # If there are more than 10 unique distances, we need to select which ones to use
            # Strategy: if more than 10, evenly sample or use first 10
            # For now, use first 10 unique distances
            if len(unique_distances) >= 10:
                # Use first 10 distances
                selected_distances = unique_distances[:10]
            else:
                # If fewer than 10, use all available
                selected_distances = unique_distances
            
            gid_mapping = {}
            num_distances = len(selected_distances)
            
            if use_reverse:
                # Reverse mapping: index 9 -> smallest distance, index 0 -> largest distance
                for idx in range(num_distances):
                    zarr_index = (num_distances - 1) - idx  # Reverse mapping: 9, 8, 7, ..., 0
                    gid_mapping[zarr_index] = selected_distances[idx]
            else:
                # Direct mapping: index 0 -> smallest distance, index 9 -> largest distance
                for idx in range(num_distances):
                    gid_mapping[idx] = selected_distances[idx]
            
            distance_mapping[setup][gid] = gid_mapping
            
            mapping_type = "reverse" if use_reverse else "direct"
            print(f"  Setup {setup}, GID {gid}: {len(gid_mapping)} distance points mapped ({mapping_type}, range: {min(selected_distances):.3f} to {max(selected_distances):.3f} m)")
    
    return distance_mapping


def consolidate_s3_zarr_results(br_data, s3_bucket_path="oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0"):
    """
    Consolidate agrivoltaics irradiance data from S3 zarr files.
    Uses GIDs from bifacial radiance data and maps distance indices to actual distances.
    
    Parameters
    ----------
    br_data : pd.DataFrame
        Bifacial radiance data (used to get GIDs and distance mappings)
    s3_bucket_path : str, default "oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0"
        S3 path to the zarr files directory
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front, data_source
    """
    # Get unique GIDs from bifacial radiance data
    validation_gids = sorted(br_data['gid'].unique())
    print(f"\nUsing {len(validation_gids)} GIDs from bifacial radiance data: {validation_gids}")
    
    # Create distance mapping
    distance_mapping = create_distance_mapping(br_data)
    
    # Initialize list to store data from each setup
    all_data = []
    
    # Process each setup present in the BR validation outputs
    for setup_num in sorted(br_data['setup'].unique()):
        print(f"\nProcessing setup {setup_num}...")
        
        # Construct zarr file path
        zarr_filename = f"configuration_{setup_num:02d}.zarr"
        zarr_path = f"s3://{s3_bucket_path}/{zarr_filename}"
        
        print(f"  Opening: {zarr_path}")
        
        try:
            # Create fsspec mapper for S3
            mapper = fsspec.get_mapper(zarr_path, anon=True)
            
            # Open zarr dataset
            ds = xr.open_zarr(mapper)
            
            print(f"  Dataset dimensions: {dict(ds.sizes)}")
            
            # Check if ground_irradiance exists
            if 'ground_irradiance' not in ds.data_vars:
                warnings.warn(f"ground_irradiance not found in setup {setup_num}")
                continue
            
            # Get ground_irradiance data
            ground_irr = ds['ground_irradiance']
            
            # Get GIDs from the dataset
            dataset_gids = ds['gid'].values
            
            # Find which validation GIDs are in this dataset
            gid_mask = np.isin(dataset_gids, validation_gids)
            matching_gids = dataset_gids[gid_mask]
            
            if len(matching_gids) == 0:
                print(f"  No matching GIDs found in setup {setup_num}")
                continue
            
            print(f"  Found {len(matching_gids)} matching GIDs")
            
            # Get indices of matching GIDs
            gid_indices = np.where(gid_mask)[0]
            
            # Select data for matching GIDs
            selected_data = ground_irr.isel(gid=gid_indices)
            
            # Convert to DataFrame
            df = selected_data.to_dataframe(name='Wm2Front').reset_index()
            
            # Map distance indices to actual distance values
            df['x'] = df.apply(
                lambda row: distance_mapping.get(setup_num, {}).get(row['gid'], {}).get(row['distance'], np.nan),
                axis=1
            )
            
            # Drop rows where distance mapping failed
            df = df.dropna(subset=['x'])
            
            # Add metadata columns
            df['setup'] = setup_num
            df['data_source'] = 'pysam'
            
            # Reorder columns: gid, setup, datetime (time), x, Wm2Front, data_source
            df = df[['gid', 'setup', 'time', 'x', 'Wm2Front', 'data_source']]
            
            # Rename time to datetime
            df = df.rename(columns={'time': 'datetime'})
            
            # Convert datetime to pandas datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Adjust timestamp: change year to 2023, retaining month, day, and hour
            df['datetime'] = df['datetime'].apply(
                lambda dt: dt.replace(year=2023)
            )
            
            # Convert gid to int
            df['gid'] = df['gid'].astype(int)
            
            # Convert x to float
            df['x'] = df['x'].astype(float)
            
            # Convert Wm2Front to float
            df['Wm2Front'] = df['Wm2Front'].astype(float)
            
            # Remove any NaN values
            df = df.dropna(subset=['Wm2Front'])
            
            print(f"  Extracted {len(df)} rows")
            
            all_data.append(df)
            
        except Exception as e:
            warnings.warn(f"Error processing setup {setup_num}: {e}")
            continue
    
    # Combine all data
    if len(all_data) == 0:
        raise ValueError("No zarr data loaded. Check S3 access and zarr file structure.")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nLoaded {len(combined_data)} rows of zarr data")
    print(f"Found {combined_data['gid'].nunique()} GIDs")
    print(f"Found {combined_data['setup'].nunique()} setups")
    print(f"Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
    
    return combined_data


def consolidate_all_results(folder_name, base_path=".", s3_bucket_path="oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0"):
    """
    Consolidate both bifacial radiance and S3 zarr results into a single DataFrame.
    
    Parameters
    ----------
    folder_name : str
        Name of the validation_results folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the validation_results folder. Defaults to current directory.
    s3_bucket_path : str, default "oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0"
        S3 path to the zarr files directory
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front, data_source
        Contains data from both bifacial_radiance and pysam sources.
        Output `setup` values follow current S3 configuration numbering.
    """
    print("="*60)
    print("CONSOLIDATING ALL RESULTS")
    print("="*60)
    
    # First, consolidate bifacial radiance results
    print("\n" + "="*60)
    print("STEP 1: Processing Bifacial Radiance Results")
    print("="*60)
    br_data = consolidate_br_results(folder_name, base_path=base_path)
    br_data = remap_validation_setups_to_s3(br_data)
    
    # Then, consolidate S3 zarr results using the same GIDs
    print("\n" + "="*60)
    print("STEP 2: Processing S3 Zarr Results")
    print("="*60)
    zarr_data = consolidate_s3_zarr_results(br_data, s3_bucket_path=s3_bucket_path)
    
    # Combine both datasets
    print("\n" + "="*60)
    print("STEP 3: Combining Datasets")
    print("="*60)
    all_data = pd.concat([br_data, zarr_data], ignore_index=True)
    
    # Sort the data
    final_data = all_data.sort_values(['data_source', 'gid', 'setup', 'datetime', 'x']).reset_index(drop=True)
    
    print(f"\nTotal rows: {len(final_data)}")
    print(f"Bifacial Radiance rows: {len(br_data)}")
    print(f"PySAM (zarr) rows: {len(zarr_data)}")
    print(f"Unique GIDs: {final_data['gid'].nunique()}")
    print(f"Unique setups: {final_data['setup'].nunique()}")
    print(f"Setup IDs in final data (S3 numbering): {sorted(final_data['setup'].unique())}")
    print(f"Data sources: {final_data['data_source'].unique()}")
    
    return final_data


def main():
    """Command-line interface for consolidating all results."""
    parser = argparse.ArgumentParser(
        description='Consolidate bifacial radiance and S3 zarr results into a single CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consolidate_all_results.py validation_results
  python consolidate_all_results.py validation_results --base-path /path/to/folder
  python consolidate_all_results.py validation_results --output all_results.pkl
  python consolidate_all_results.py validation_results --br-only-full-res
        """
    )
    
    parser.add_argument(
        'folder_name',
        type=str,
        help='Name of the validation_results folder (e.g., "validation_results")'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='.',
        help='Base path to the validation_results folder (defaults to current directory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle file name (defaults to "all_results.pkl" or "br_full_resolution_results.pkl" if --br-only-full-res is used)'
    )
    
    parser.add_argument(
        '--s3-path',
        type=str,
        default='oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0',
        help='S3 path to zarr files directory (default: oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0)'
    )
    
    parser.add_argument(
        '--br-only-full-res',
        action='store_true',
        help='Consolidate only bifacial radiance results at full resolution (100 points) without SAM results'
    )
    
    args = parser.parse_args()
    
    # Handle BR-only full resolution mode
    if args.br_only_full_res:
        # Set default output filename if not provided
        output_file = args.output if args.output else 'br_full_resolution_results.pkl'
        
        print("Consolidating bifacial radiance results at full resolution...")
        print(f"Validation results folder: {args.folder_name}")
        print(f"Base path: {args.base_path}")
        print(f"Output file: {output_file}\n")
        
        # Consolidate BR data at full resolution
        data = consolidate_br_results_full_resolution(
            args.folder_name,
            base_path=args.base_path
        )
        
        # Write to pickle
        print(f"\nWriting data to {output_file}...")
        data.to_pickle(output_file)
        
        file_size = Path(output_file).stat().st_size
        print(f"Done! Data exported to {output_file}")
        print(f"File size: {file_size:,} bytes")
        return
    
    # Default behavior: consolidate all results
    # Set default output filename if not provided
    output_file = args.output if args.output else 'all_results.pkl'
    
    print("Consolidating all results...")
    print(f"Validation results folder: {args.folder_name}")
    print(f"Base path: {args.base_path}")
    print(f"S3 path: {args.s3_path}")
    print(f"Output file: {output_file}\n")
    
    # Consolidate the data
    data = consolidate_all_results(
        args.folder_name,
        base_path=args.base_path,
        s3_bucket_path=args.s3_path
    )
    
    # Write to pickle
    print(f"\nWriting data to {output_file}...")
    data.to_pickle(output_file)
    
    file_size = Path(output_file).stat().st_size
    print(f"Done! Data exported to {output_file}")
    print(f"File size: {file_size:,} bytes")


if __name__ == "__main__":
    main()

