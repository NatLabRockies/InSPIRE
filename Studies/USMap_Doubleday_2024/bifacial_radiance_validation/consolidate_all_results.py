"""
Consolidate All Results Script
This script consolidates both bifacial radiance validation data and S3 zarr data
into a single CSV file with a data_source column indicating the source.

First, it processes bifacial radiance results from the validation_results folder.
Then, it uses the same GIDs to access corresponding results from S3 zarr files.
The zarr distance indices (0-9) are mapped to actual distance values in meters
from the bifacial radiance data.

Usage as Python function:
    from consolidate_all_results import consolidate_all_results
    data = consolidate_all_results("validation_results", 
                                   base_path="bifacial_radiance_validation")

Usage from command line:
    python consolidate_all_results.py validation_results
    python consolidate_all_results.py validation_results --base-path "/path/to/folder"
    python consolidate_all_results.py validation_results --output all_results.pkl
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
        Contains only timestamps that exist in the CSV files.
    """
    # Construct full path to folder
    folder_path = Path(base_path) / folder_name
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all setup folders (should be numbered 1-10)
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
                        
                        # Add metadata columns
                        csv_data['gid'] = int(gid)
                        csv_data['setup'] = setup_num
                        csv_data['datetime'] = file_datetime
                        csv_data['data_source'] = 'bifacial_radiance'
                        
                        # Reorder columns
                        csv_data = csv_data[['gid', 'setup', 'datetime', 'x', 'Wm2Front', 'data_source']]
                        
                        all_data.append(csv_data)
                        
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


def create_distance_mapping(br_data):
    """
    Create a mapping from zarr distance indices (0-9) to actual distance values in meters.
    Uses the distance values from bifacial radiance data, which are monotonically increasing.
    Maps index 9 to the smallest distance, index 8 to the next smallest, ..., index 0 to the largest.
    
    Parameters
    ----------
    br_data : pd.DataFrame
        Bifacial radiance data with 'x' column containing distance values
    
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
        
        for gid in sorted(setup_data['gid'].unique()):
            gid_data = setup_data[setup_data['gid'] == gid]
            
            # Get unique distance values and sort them (monotonically increasing)
            unique_distances = sorted(gid_data['x'].unique())
            
            # Map indices 0-9 to distance values (reversed: index 9 maps to smallest distance)
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
            # Map in reverse: index 9 -> smallest distance, index 0 -> largest distance
            for idx in range(num_distances):
                zarr_index = (num_distances - 1) - idx  # Reverse mapping: 9, 8, 7, ..., 0
                gid_mapping[zarr_index] = selected_distances[idx]
            
            distance_mapping[setup][gid] = gid_mapping
            
            print(f"  Setup {setup}, GID {gid}: {len(gid_mapping)} distance points mapped (range: {min(selected_distances):.3f} to {max(selected_distances):.3f} m)")
    
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
    
    # Process each setup (1-10)
    for setup_num in range(1, 11):
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
    """
    print("="*60)
    print("CONSOLIDATING ALL RESULTS")
    print("="*60)
    
    # First, consolidate bifacial radiance results
    print("\n" + "="*60)
    print("STEP 1: Processing Bifacial Radiance Results")
    print("="*60)
    br_data = consolidate_br_results(folder_name, base_path=base_path)
    
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
        help='Output pickle file name (defaults to "all_results.pkl")'
    )
    
    parser.add_argument(
        '--s3-path',
        type=str,
        default='oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0',
        help='S3 path to zarr files directory (default: oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0)'
    )
    
    args = parser.parse_args()
    
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

