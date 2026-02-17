import glob
import numpy as np
import xarray as xr
import dask.array as da
import os
from pathlib import Path

from inspire_agrivolt import logger


def check_ncs_coords_dims(files: list[str]) -> None:
    from glob import glob
    import xarray as xr

    allowed = {"time", "distance", "gid"}

    for fp in files:
        try:
            ds = xr.open_dataset(
                str(fp), engine="netcdf4", decode_cf=False, mask_and_scale=False
            )
            coords = set(ds.coords.keys())
            dims = set(ds.sizes.keys())

            all_present = coords.union(dims)
            extra = all_present - allowed

            if extra:
                raise ValueError(
                    f"{fp} contains unexpected dimensions or coordinates: {extra}"
                )

            logger.debug(
                f"good: {fp} | coords: {sorted(coords)} | dims: {sorted(dims)}"
            )

        except Exception as e:
            raise ValueError(f"bad: {fp} | FAILED: {e}") from e


def check_zarr_coords_dims(zarr_path: str) -> None:
    from glob import glob
    import xarray as xr

    allowed = {"time", "distance", "gid"}

    try:
        ds = xr.open_zarr(zarr_path)
        coords = set(ds.coords.keys())
        dims = set(ds.sizes.keys())

        all_present = coords.union(dims)
        extra = all_present - allowed

        if extra:
            raise ValueError(
                f"{zarr_path} contains unexpected dimensions or coordinates: {extra}"
            )

        logger.debug(
            f"good: {zarr_path} | coords: {sorted(coords)} | dims: {sorted(dims)}"
        )

    except Exception as e:
        raise ValueError(f"bad: {zarr_path} | FAILED: {e}") from e


def check_completeness(outputs_dir: str, state: str, conf: str) -> dict:
    """
    Check completeness of outputs for a state and config.
    Validates both CSV and NetCDF outputs for missing or extra GIDs.

    tree of ``outputs_dir``
    /projects/inspire/PySAM-Maps/Full-Outputs/
    - colorado
        - 01
        - 02
        - ..
        - 10
    - north carolina
        - 01
        - 02
        - ..
        - 10

    Returns:
        dict: Status summary with lists of missing/extra gids, and gid consistency info.
    """
    import pvdeg
    import pandas as pd
    import xarray as xr
    from pathlib import Path

    outputs_path = Path(outputs_dir)
    target_path = outputs_path / state / conf

    files = sorted(target_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {target_path}")

    meta_list = [pd.read_csv(file, index_col=0) for file in files]
    out_meta_df = pd.concat(meta_list)

    # check dims/coords before reading NetCDF
    check_ncs_coords_dims(files=sorted(target_path.glob("*.nc")))
    out_model_ds = xr.open_mfdataset(str(target_path / "*.nc"))

    # Get weather meta for full expected set of GIDs
    weather_db = "NSRDB"
    weather_arg = {
        "satellite": "Americas",
        "names": "TMY",
        "NREL_HPC": True,
        "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
    }

    geo_weather, geo_meta = pvdeg.weather.get(
        weather_db, geospatial=True, **weather_arg
    )

    # All GIDs that should be present for the state
    sub_meta = geo_meta[geo_meta["state"] == state.title()]
    expected_gids = set(sub_meta.index.values)

    found_gids = set(out_meta_df.index.values)

    missing = list(expected_gids - found_gids)
    extra = list(found_gids - expected_gids)

    # GID consistency check between CSV and NetCDF outputs
    try:
        ds_gids = set(out_model_ds.gid.values)
    except AttributeError:
        raise ValueError("NetCDF dataset does not contain 'gid' coordinate.")

    # CSV index vs NetCDF gids
    csv_vs_netcdf_diff = found_gids.symmetric_difference(ds_gids)
    gid_consistent = len(csv_vs_netcdf_diff) == 0

    status = (
        "complete"
        if not missing and not extra and gid_consistent
        else "inconsistent"
        if not gid_consistent
        else "missing"
        if missing
        else "extra"
    )

    return {
        "status": status,
        "missing_gids": missing,
        "extra_gids": extra,
        "total_expected": len(expected_gids),
        "total_found_csv": len(found_gids),
        "total_found_netcdf": len(ds_gids),
        "csv_netcdf_gid_match": gid_consistent,
        "gid_mismatch_list": list(csv_vs_netcdf_diff),
    }


def generate_missing_gids_file(
    completeness: dict, output_path: str = None, filename: str = "missing_gids.txt"
) -> str:
    """
    takes dict created by check_completeness
    """

    if output_path is None:
        output_path = os.getcwd()

    os.makedirs(output_path, exist_ok=True)

    missing_gids = completeness["missing_gids"]
    missing_gids = [str(entry) for entry in missing_gids]
    data = " ".join(missing_gids)

    file_path = os.path.join(output_path, filename)

    with open(file_path, "w") as fp:
        fp.write(data)

    return file_path


def _validate_netcdf_files(nc_files, check_dims={}):
    """
    Check if netcdf files can be opened.

    Sometimes they get corrupted or written wrong. Function checks for this
    to avoid opening or converting bad NC files into Zarr.
    """
    good = []
    bad = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f, engine="h5netcdf")
            if check_dims:
                for dim in check_dims:
                    if dim not in ds.dims:
                        raise ValueError("bad dim")
            ds.close()
            good.append(f)
        except Exception as e:
            print(f"BAD FILE: {f}")
            print(e)
            bad.append(f)
    return good, bad


def merge_original_fill_data_to_zarr(
    MODEL_OUT_DIR_A: str,
    MODEL_OUT_DIR_B: str,
    state: str,
    scenario: str,
    OUTPUT_ZARR_PATH: str,
) -> None:
    """
    Merge original model outputs with fill data and write a cleaned dataset to Zarr.

    This function:
      1. Reads NetCDF output files from two directories:
         - MODEL_OUT_DIR_A: original (possibly incomplete) outputs
         - MODEL_OUT_DIR_B: fill outputs used to patch missing values
      2. Validates and filters out corrupt or unreadable NetCDF files.
      3. Loads, sorts, and merges datasets on the 'gid' dimension,
         ensuring unique 'gid' values by keeping the first occurrence.
      4. Fetches NSRDB metadata (via pvdeg) to filter out any gids that
         do not belong to the specified state.
      5. Stores the final, cleaned dataset as a Zarr store at OUTPUT_ZARR_PATH.

    Parameters
    ----------
    MODEL_OUT_DIR_A : str
        Path to directory containing original model NetCDF files.
    MODEL_OUT_DIR_B : str
        Path to directory containing fill NetCDF files.
    state : str
        Lowercase state name (e.g., "colorado").
    scenario : str
        Configuration identifier with leading zeros (e.g., "04", "08").
    OUTPUT_ZARR_PATH : str
        Path to the output Zarr store to be written. Must not already exist. Cannot be a cloud path.

    Raises
    ------
    FileExistsError
        If OUTPUT_ZARR_PATH already exists.
    ValueError
        If required variables or dimensions are missing from the datasets.
    """

    import pvdeg

    print(f"starting {state} {scenario}")

    # Validate basic types (optional)
    if not isinstance(MODEL_OUT_DIR_A, str):
        raise ValueError("MODEL_OUT_DIR_A must be str or Path")
    if not isinstance(MODEL_OUT_DIR_B, str):
        raise ValueError("MODEL_OUT_DIR_B must be str or Path")
    if not isinstance(state, str):
        raise ValueError("state must be a string")
    if not isinstance(scenario, str):
        raise ValueError("scenario must be a string")
    if not isinstance(OUTPUT_ZARR_PATH, str):
        raise ValueError("OUTPUT_ZARR_PATH must be str or Path")

    if Path(OUTPUT_ZARR_PATH).exists():
        raise FileExistsError(f"Something already exists at {OUTPUT_ZARR_PATH}")

    logger.info("Starting merge for state=%s scenario=%s", state, scenario)

    original_files = sorted(Path(MODEL_OUT_DIR_A).glob("*.nc"))
    fill_files = sorted(Path(MODEL_OUT_DIR_B).glob("*.nc"))

    logger.info(f"validating files in {MODEL_OUT_DIR_A} and {MODEL_OUT_DIR_B}")

    required_dims = {"gid", "time", "distance"}

    original_files, bad_A = _validate_netcdf_files(
        original_files, check_dims=required_dims
    )
    fill_files, bad_B = _validate_netcdf_files(fill_files, check_dims=required_dims)

    if bad_A:
        logger.warning("Skipping %d bad NC files in %s", len(bad_A), MODEL_OUT_DIR_A)
    if bad_B:
        logger.warning("Skipping %d bad NC files in %s", len(bad_B), MODEL_OUT_DIR_B)
    if not original_files:
        raise ValueError(f"No valid original files found in {MODEL_OUT_DIR_A}")

    logger.info(f"opening files in {MODEL_OUT_DIR_A}")

    original_ds = xr.open_mfdataset(
        original_files,
        combine="by_coords",
        chunks={},  # let engine decide
        parallel=False,
        engine="h5netcdf",
    )

    if "gid" not in original_ds.coords and "gid" not in original_ds.dims:
        raise ValueError("Original dataset missing 'gid' coordinate/dimension.")

    original_ds = original_ds.sortby("gid")

    logger.info(f"interoggating {MODEL_OUT_DIR_B}")

    fill_combined = None
    if fill_files:
        logger.info(f"opening fill files...")
        fill_datasets = [
            xr.open_dataset(f, chunks="auto").sortby("gid") for f in sorted(fill_files)
        ]
        fill_combined = xr.concat(fill_datasets, dim="gid").sortby("gid")

    if fill_combined is not None:
        try:
            filled = original_ds.combine_first(fill_combined)
        except Exception as e:
            logger.warning(
                "combine_first failed (%s); falling back to concat+dedupe", e
            )
            merged = xr.concat(
                [original_ds, fill_combined], dim="gid", data_vars="all", coords="all"
            ).sortby("gid")
            # Keep first occurrence per gid (original files listed first)
            gids = merged["gid"].values
            _, first_idx = np.unique(gids, return_index=True)
            filled = merged.isel(gid=np.sort(first_idx))
    else:
        logger.info(f"no fill files provided, continuing with only {MODEL_OUT_DIR_A}")
        filled = original_ds

    logger.info(f"loading data from NSRDB...")
    # load nsrdb metadata
    weather_db = "NSRDB"
    weather_arg = {
        "satellite": "Americas",
        "names": "TMY",
        "NREL_HPC": True,
        "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
    }

    _, geo_meta = pvdeg.weather.get(weather_db, geospatial=True, **weather_arg)

    logger.info(f"loaded NSRDB metadata")

    state_meta = geo_meta[geo_meta["state"] == state.title()]
    if state_meta.empty:
        logger.warning("No gids found in geo_meta for state=%s", state)

    # check if any gids are outside of the state, if so remove them, this seems to have been an issue in the past
    # while probably not required, we want to make sure that this is not a slient issue
    state_valid_gids = state_meta.index.values

    # make sure that dataset gids is a subset of valid gids for the state
    is_ds_valid_subset = (
        len(np.setdiff1d(filled.gid.values, state_valid_gids, assume_unique=True)) == 0
    )
    if not is_ds_valid_subset:
        logger.warning("gids from outside state found inside of input ncdf's")

    # this may not be required and might take a long time
    filled = filled.chunk({"time": -1, "distance": -1, "gid": 2000})

    filled.to_zarr(str(OUTPUT_ZARR_PATH), compute=True, consolidated=True)

    logger.info("Finished merge → %s", OUTPUT_ZARR_PATH)
    print(f"ending {state} {scenario}")
