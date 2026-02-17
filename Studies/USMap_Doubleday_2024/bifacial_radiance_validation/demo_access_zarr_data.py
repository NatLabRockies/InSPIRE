"""
Demo script for accessing agrivoltaics irradiance data from S3 zarr files.

The demo can be run directly (python demo_access_zarr_data.py) to see printed output
of all examples, or individual functions can be imported and used in other scripts

This script demonstrates three common use cases:
1. Loading data for selected setups by GID
2. Using the gid-lat-lon.csv lookup table to find the nearest GID for a given lat/lon
3. Accessing data for a lat/lon range

The data is stored in zarr format on S3 at:
    s3://oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0/

Each setup has its own zarr file named: configuration_{setup:02d}.zarr
"""

import pandas as pd
import xarray as xr
import fsspec
import numpy as np
from scipy.spatial.distance import cdist


# S3 bucket configuration
S3_BUCKET_PATH = "oedi-data-lake/inspire/agrivoltaics_irradiance/v1.0"
LOOKUP_TABLE_PATH = f"s3://{S3_BUCKET_PATH}/gid-lat-lon.csv"


def load_lookup_table():
    """
    Load the GID to lat/lon lookup table from S3.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, latitude, longitude
    """
    df = pd.read_csv(LOOKUP_TABLE_PATH, index_col=0)
    
    # Reset index to make GID a column
    df = df.reset_index(names='gid')
    
    return df


def open_zarr_dataset(setup_num, s3_bucket_path=S3_BUCKET_PATH):
    """
    Open a zarr dataset for a specific setup from S3.
    
    Parameters
    ----------
    setup_num : int
        Setup number (1-10)
    s3_bucket_path : str
        S3 path to the zarr files directory
    
    Returns
    -------
    xr.Dataset
        Opened xarray dataset
    """
    zarr_filename = f"configuration_{setup_num:02d}.zarr"
    zarr_path = f"s3://{s3_bucket_path}/{zarr_filename}"
    
    # Create fsspec mapper for S3 (anonymous access)
    mapper = fsspec.get_mapper(zarr_path, anon=True)
    
    # Open zarr dataset
    ds = xr.open_zarr(mapper)
    
    return ds


def load_data_by_gid(setup_num, gids, s3_bucket_path=S3_BUCKET_PATH):
    """
    Load data for specific GIDs from a setup.
    
    Parameters
    ----------
    setup_num : int
        Setup number (1-10)
    gids : list of int
        List of GIDs to load
    s3_bucket_path : str
        S3 path to the zarr files directory
    
    Returns
    -------
    xr.Dataset
        Dataset subset containing only the specified GIDs, or None if no matching GIDs found
    list
        List of matching GIDs found in the dataset
    """
    # Open the zarr dataset
    ds = open_zarr_dataset(setup_num, s3_bucket_path)
    
    # Get all GIDs in the dataset
    dataset_gids = ds['gid'].values
    
    # Find which requested GIDs are in this dataset
    gid_mask = np.isin(dataset_gids, gids)
    matching_gids = dataset_gids[gid_mask].tolist()
    
    if len(matching_gids) == 0:
        return None, []
    
    # Get indices of matching GIDs
    gid_indices = np.where(gid_mask)[0]
    
    # Select data for matching GIDs
    selected_data = ds.isel(gid=gid_indices)
    
    return selected_data, matching_gids


def load_data_by_gid_multiple_setups(setup_nums, gids, s3_bucket_path=S3_BUCKET_PATH):
    """
    Load data for specific GIDs from multiple setups and combine them.
    
    Parameters
    ----------
    setup_nums : list of int
        List of setup numbers (1-10)
    gids : list of int
        List of GIDs to load
    s3_bucket_path : str
        S3 path to the zarr files directory
    
    Returns
    -------
    xr.Dataset
        Combined dataset with a 'setup' dimension, or None if no matching GIDs found
    dict
        Dictionary mapping setup numbers to lists of matching GIDs found in each dataset
    """
    datasets = []
    matching_gids_dict = {}
    
    for setup_num in setup_nums:
        data, matching_gids = load_data_by_gid(setup_num, gids, s3_bucket_path)
        if data is not None:
            # Add setup dimension
            data = data.expand_dims('setup')
            data = data.assign_coords(setup=[setup_num])
            datasets.append(data)
            matching_gids_dict[setup_num] = matching_gids
    
    if len(datasets) == 0:
        return None, {}
    
    # Concatenate along setup dimension
    combined_data = xr.concat(datasets, dim='setup')
    
    return combined_data, matching_gids_dict


def find_nearest_gid(latitude, longitude, lookup_df=None):
    """
    Find the nearest GID for a given latitude/longitude using nearest neighbor search.
    
    Parameters
    ----------
    latitude : float
        Target latitude
    longitude : float
        Target longitude
    lookup_df : pd.DataFrame, optional
        Lookup table DataFrame. If None, will load from S3.
    
    Returns
    -------
    int
        Nearest GID
    float
        Distance to nearest point (in degrees)
    float
        Nearest latitude
    float
        Nearest longitude
    """
    # Load lookup table if not provided
    if lookup_df is None:
        lookup_df = load_lookup_table()
    
    # Extract lat/lon coordinates
    coords = lookup_df[['latitude', 'longitude']].values
    
    # Target point
    target = np.array([[latitude, longitude]])
    
    # Calculate distances (using Euclidean distance in lat/lon space)
    distances = cdist(target, coords, metric='euclidean')[0]
    
    # Find nearest point
    nearest_idx = np.argmin(distances)
    nearest_distance = distances[nearest_idx]
    
    # Get nearest GID and coordinates
    nearest_row = lookup_df.iloc[nearest_idx]
    nearest_gid = int(nearest_row['gid'])
    nearest_lat = float(nearest_row['latitude'])
    nearest_lon = float(nearest_row['longitude'])
    
    return nearest_gid, nearest_distance, nearest_lat, nearest_lon


def load_data_by_lat_lon(latitude, longitude, setup_num, s3_bucket_path=S3_BUCKET_PATH, 
                         lookup_df=None):
    """
    Load data for a specific lat/lon by finding the nearest GID.
    
    Parameters
    ----------
    latitude : float
        Target latitude
    longitude : float
        Target longitude
    setup_num : int
        Setup number (1-10)
    s3_bucket_path : str
        S3 path to the zarr files directory
    lookup_df : pd.DataFrame, optional
        Lookup table DataFrame. If None, will load from S3.
    
    Returns
    -------
    xr.Dataset or None
        Dataset for the nearest GID, or None if GID not found
    int
        GID that was used
    float
        Distance to nearest point (in degrees)
    float
        Nearest latitude
    float
        Nearest longitude
    """
    # Find nearest GID
    nearest_gid, distance, nearest_lat, nearest_lon = find_nearest_gid(
        latitude, longitude, lookup_df=lookup_df
    )
    
    # Load data for that GID
    data, matching_gids = load_data_by_gid(setup_num, [nearest_gid], s3_bucket_path)
    
    return data, nearest_gid, distance, nearest_lat, nearest_lon


def load_data_by_lat_lon_multiple_setups(latitude, longitude, setup_nums, 
                                         s3_bucket_path=S3_BUCKET_PATH, lookup_df=None):
    """
    Load data for a specific lat/lon by finding the nearest GID, from multiple setups.
    
    Parameters
    ----------
    latitude : float
        Target latitude
    longitude : float
        Target longitude
    setup_nums : list of int
        List of setup numbers (1-10)
    s3_bucket_path : str
        S3 path to the zarr files directory
    lookup_df : pd.DataFrame, optional
        Lookup table DataFrame. If None, will load from S3.
    
    Returns
    -------
    xr.Dataset or None
        Combined dataset with a 'setup' dimension, or None if GID not found
    int
        GID that was used
    float
        Distance to nearest point (in degrees)
    float
        Nearest latitude
    float
        Nearest longitude
    """
    # Find nearest GID
    nearest_gid, distance, nearest_lat, nearest_lon = find_nearest_gid(
        latitude, longitude, lookup_df=lookup_df
    )
    
    # Load data for that GID from multiple setups
    data, matching_gids_dict = load_data_by_gid_multiple_setups(
        setup_nums, [nearest_gid], s3_bucket_path
    )
    
    return data, nearest_gid, distance, nearest_lat, nearest_lon


def load_data_by_lat_lon_range(lat_min, lat_max, lon_min, lon_max, setup_num, 
                                s3_bucket_path=S3_BUCKET_PATH, lookup_df=None):
    """
    Load data for all GIDs within a lat/lon bounding box.
    
    Parameters
    ----------
    lat_min : float
        Minimum latitude
    lat_max : float
        Maximum latitude
    lon_min : float
        Minimum longitude
    lon_max : float
        Maximum longitude
    setup_num : int
        Setup number (1-10)
    s3_bucket_path : str
        S3 path to the zarr files directory
    lookup_df : pd.DataFrame, optional
        Lookup table DataFrame. If None, will load from S3.
    
    Returns
    -------
    xr.Dataset or None
        Dataset containing all GIDs within the bounding box, or None if no GIDs found
    pd.DataFrame
        DataFrame of GIDs and their coordinates within the range
    list
        List of matching GIDs found in the dataset
    """
    # Load lookup table if not provided
    if lookup_df is None:
        lookup_df = load_lookup_table()
    
    # Find GIDs within the bounding box
    mask = (
        (lookup_df['latitude'] >= lat_min) & 
        (lookup_df['latitude'] <= lat_max) &
        (lookup_df['longitude'] >= lon_min) & 
        (lookup_df['longitude'] <= lon_max)
    )
    
    gids_in_range = lookup_df[mask]
    
    if len(gids_in_range) == 0:
        return None, None, []
    
    # Get list of GIDs
    gid_list = gids_in_range['gid'].tolist()
    
    # Load data for these GIDs
    data, matching_gids = load_data_by_gid(setup_num, gid_list, s3_bucket_path)
    
    return data, gids_in_range, matching_gids


def load_data_by_lat_lon_range_multiple_setups(lat_min, lat_max, lon_min, lon_max, setup_nums,
                                               s3_bucket_path=S3_BUCKET_PATH, lookup_df=None):
    """
    Load data for all GIDs within a lat/lon bounding box from multiple setups.
    
    Parameters
    ----------
    lat_min : float
        Minimum latitude
    lat_max : float
        Maximum latitude
    lon_min : float
        Minimum longitude
    lon_max : float
        Maximum longitude
    setup_nums : list of int
        List of setup numbers (1-10)
    s3_bucket_path : str
        S3 path to the zarr files directory
    lookup_df : pd.DataFrame, optional
        Lookup table DataFrame. If None, will load from S3.
    
    Returns
    -------
    xr.Dataset or None
        Combined dataset with a 'setup' dimension, or None if no GIDs found
    pd.DataFrame
        DataFrame of GIDs and their coordinates within the range
    dict
        Dictionary mapping setup numbers to lists of matching GIDs found in each dataset
    """
    # Load lookup table if not provided
    if lookup_df is None:
        lookup_df = load_lookup_table()
    
    # Find GIDs within the bounding box
    mask = (
        (lookup_df['latitude'] >= lat_min) & 
        (lookup_df['latitude'] <= lat_max) &
        (lookup_df['longitude'] >= lon_min) & 
        (lookup_df['longitude'] <= lon_max)
    )
    
    gids_in_range = lookup_df[mask]
    
    if len(gids_in_range) == 0:
        return None, None, {}
    
    # Get list of GIDs
    gid_list = gids_in_range['gid'].tolist()
    
    # Load data for these GIDs from multiple setups
    data, matching_gids_dict = load_data_by_gid_multiple_setups(
        setup_nums, gid_list, s3_bucket_path
    )
    
    return data, gids_in_range, matching_gids_dict


def main():
    """
    Main demo function showing all three use cases.
    """
    print("="*60)
    print("DEMO: Accessing Agrivoltaics Irradiance Data from S3 Zarr Files")
    print("="*60)
    
    # Use multiple setups for all examples
    setup_nums = [1, 2, 3]
    
    # Example 1: Load data by GID
    print("\n" + "="*60)
    print("EXAMPLE 1: Loading data for specific GIDs from multiple setups")
    print("="*60)
    
    example_gids = [243498, 886847]
    print(f"Setups: {setup_nums}")
    print(f"Requested GIDs: {example_gids}")
    
    for setup_num in setup_nums:
        zarr_path = f"s3://{S3_BUCKET_PATH}/configuration_{setup_num:02d}.zarr"
        print(f"Opening zarr file: {zarr_path}")
    
    data_by_gid, matching_gids_dict = load_data_by_gid_multiple_setups(
        setup_nums, example_gids
    )
    
    if data_by_gid is None:
        print(f"WARNING: None of the requested GIDs {example_gids} are in these datasets")
    else:
        print(f"\nMatching GIDs by setup:")
        for setup_num, matching_gids in matching_gids_dict.items():
            print(f"  Setup {setup_num}: {matching_gids}")
        print(f"\nCombined dataset dimensions: {dict(data_by_gid.sizes)}")
        print(f"Data variables: {list(data_by_gid.data_vars)}")
        
        if 'ground_irradiance' in data_by_gid.data_vars:
            print(f"Ground irradiance shape: {data_by_gid['ground_irradiance'].shape}")
            print(f"Ground irradiance dimensions: {data_by_gid['ground_irradiance'].dims}")
            print("\nSample of ground_irradiance data (setup=1, time=0, distance=0):")
            print(data_by_gid['ground_irradiance'].isel(setup=0, time=0, distance=0))
    
    # Example 2: Find nearest GID for a lat/lon and load data
    print("\n" + "="*60)
    print("EXAMPLE 2: Finding nearest GID for lat/lon and loading data from multiple setups")
    print("="*60)
    
    target_lat = 39.7392
    target_lon = -104.9903
    print(f"Target location: ({target_lat}, {target_lon})")
    print(f"Setups: {setup_nums}")
    print(f"Loading lookup table from: {LOOKUP_TABLE_PATH}")
    
    lookup_df = load_lookup_table()
    
    data_by_latlon, nearest_gid, distance, nearest_lat, nearest_lon = load_data_by_lat_lon_multiple_setups(
        target_lat, target_lon, setup_nums, lookup_df=lookup_df
    )
    
    print(f"\nNearest GID: {nearest_gid}")
    print(f"Nearest location: ({nearest_lat:.4f}, {nearest_lon:.4f})")
    
    if data_by_latlon is not None:
        print(f"Combined dataset dimensions: {dict(data_by_latlon.sizes)}")
        print(f"Setups included: {data_by_latlon.setup.values.tolist()}")
    
    # Example 3: Load data for a lat/lon range
    print("\n" + "="*60)
    print("EXAMPLE 3: Loading data for a lat/lon bounding box from multiple setups")
    print("="*60)
    
    colorado_lat_min = 37.0
    colorado_lat_max = 41.0
    colorado_lon_min = -109.0
    colorado_lon_max = -102.0
    print(f"Latitude range: [{colorado_lat_min}, {colorado_lat_max}]")
    print(f"Longitude range: [{colorado_lon_min}, {colorado_lon_max}]")
    print(f"Setups: {setup_nums}")
    
    data_by_range, gids_in_range, matching_gids_dict = load_data_by_lat_lon_range_multiple_setups(
        colorado_lat_min, colorado_lat_max,
        colorado_lon_min, colorado_lon_max,
        setup_nums, lookup_df=lookup_df
    )
    
    if data_by_range is None:
        print("WARNING: No GIDs found in the specified range")
    else:
        print(f"\nFound {len(gids_in_range)} GIDs in range")
        print(f"Matching GIDs by setup:")
        for setup_num, matching_gids in matching_gids_dict.items():
            print(f"  Setup {setup_num}: {len(matching_gids)} GIDs")
        print("\nFirst 10 GIDs in range:")
        print(gids_in_range[['gid', 'latitude', 'longitude']].head(10))
        if len(gids_in_range) > 10:
            print(f"... and {len(gids_in_range) - 10} more")
        
        if 'ground_irradiance' in data_by_range.data_vars:
            print(f"\nCombined data shape: {dict(data_by_range.sizes)}")
            print(f"Setups included: {data_by_range.setup.values.tolist()}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nTips:")
    print("- You can access different setups (1-10) by changing setup_nums list")
    print("- The zarr files contain multiple data variables (check ds.data_vars)")
    print("- Use xarray's selection methods (sel, isel) to subset by setup, time, distance, etc.")
    print("- Example: data.sel(setup=1) to get data for a specific setup")

if __name__ == "__main__":
    main()

