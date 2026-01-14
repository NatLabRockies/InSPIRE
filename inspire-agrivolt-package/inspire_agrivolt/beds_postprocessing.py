import numpy as np
import xarray as xr
import dask.array as da

from typing import Literal
from collections.abc import Callable, Mapping
from pathlib import Path

from inspire_agrivolt import logger

# placeholder, we never read these values from the configs
# we should place some restrictions on these to make sure they are never referenced in their current form
pitch_temp, pitchfactor, tilt = -999, -999, -999

TRACKING_SCENARIOS = {"01", "02", "03", "04", "05"}
TRACKING_3_BEDS_SCENARIOS = {"01", "02", "03", "04"}
TRACKING_6_BEDS_SCENARIOS = {"05"}
VARIABLE_PITCH_SCENARIOS = {"06", "07", "08", "09", "11"}
VERTICAL_SCENARIOS = {"10"}

ALL_SCENARIOS = TRACKING_SCENARIOS | VARIABLE_PITCH_SCENARIOS | VERTICAL_SCENARIOS


ENERGY_VARIABLES = [
    "subarray1_dc_gross",  # kWh
    "annual_energy",  # kWh
]


# these were taken from debug martin method
configs = {
    # single axis tracking
    "01": {
        "hub_height": 1.5,
        "pitch": 5,
        "sazm": 180,  # Tracker axis azimuth
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    },
    # single axis tracking
    "02": {
        "hub_height": 2.4,
        "pitch": 5,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    },  # single axis tracking
    "03": {
        "hub_height": 2.4,
        "pitch": 5,
        "sazm": 180,
        "modulename": "PVmodule_1mxgap",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    },
    "04": {
        "hub_height": 1.5,
        "pitch": 8,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 6,
        "fixed_tilt_angle": None,
    },
    "05": {
        "hub_height": 1.5,
        "pitch": 11,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 9,
        "fixed_tilt_angle": None,
    },
    "06": {
        "hub_height": 1.5,
        "tilt": None,  # fixed,
        "sazm": 180,
        "pitchfactor": 1,
        "modulename": "PVmodule",
        "pitch": pitch_temp * pitchfactor,
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "07": {
        "hub_height": 2.4,
        "sazm": 180,
        "pitchfactor": 1,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "08": {
        "hub_height": 2.4,
        "sazm": 180,
        "pitchfactor": 1,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule_1mxgap",
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "09": {
        "hub_height": 1.5,
        "sazm": 180,
        "pitchfactor": 2,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule",
        "bedsWanted": 6,
        "fixed_tilt_angle": tilt,
    },
    "10": {  # does this want 6 or 7 beds
        "hub_height": 2,
        "sazm": 90,
        "pitch": 8.6,
        "modulename": "PVmodule",
        "bedsWanted": 7,
        "xp": 8,
        "fixed_tilt_angle": 90,
    },
}


def tracking_3_beds(dataset: xr.Dataset) -> xr.Dataset:
    """
    single axis tracking system with 3 beds.

    # SCENARIO 01 02 03 04 #

    Hardcoded from information in "SETUP Description Complete.xlx" values calculated with cw = 2
    """
    left_underpanel_slice = slice(0, 3)
    right_underpanel_slice = slice(7, 10)

    bedA_slice = slice(3, 4)
    bedB_slice = slice(4, 6)
    bedC_slice = slice(6, 7)
    edgetoedge_slice = slice(3, 7)

    underpanel_left = dataset.ground_irradiance.isel(
        {"distance": left_underpanel_slice}
    ).mean("distance")
    underpanel_right = dataset.ground_irradiance.isel(
        {"distance": right_underpanel_slice}
    ).mean("distance")
    underpanel = ((underpanel_left + underpanel_right) / 2).rename("underpanel")

    bedA = (
        dataset.ground_irradiance.isel({"distance": bedA_slice})
        .mean("distance")
        .rename("beda")
    )
    bedB = (
        dataset.ground_irradiance.isel({"distance": bedB_slice})
        .mean("distance")
        .rename("bedb")
    )
    bedC = (
        dataset.ground_irradiance.isel({"distance": bedC_slice})
        .mean("distance")
        .rename("bedc")
    )

    edgetoedge = (
        dataset.ground_irradiance.isel({"distance": edgetoedge_slice})
        .mean("distance")
        .rename("edgetoedge")
    )

    beds_ds = xr.merge([underpanel, edgetoedge, bedA, bedB, bedC])

    return beds_ds


def tracking_6_beds(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    single axis tracking system with 6 beds

    # SCENARIO 5 #
    """

    left_underpanel_slice = slice(0, 2)
    right_underpanel_slice = slice(8, 10)

    bedA_slice = slice(2, 3)
    bedB_slice = slice(3, 4)
    bedC_slice = slice(4, 5)
    bedD_slice = slice(5, 6)
    bedE_slice = slice(6, 7)
    bedF_slice = slice(7, 8)
    edgetoedge_slice = slice(2, 8)

    underpanel_left = dataset.ground_irradiance.isel(
        {"distance": left_underpanel_slice}
    ).mean("distance")
    underpanel_right = dataset.ground_irradiance.isel(
        {"distance": right_underpanel_slice}
    ).mean("distance")
    underpanel = ((underpanel_left + underpanel_right) / 2).rename("underpanel")

    bedA = (
        dataset.ground_irradiance.isel({"distance": bedA_slice})
        .mean("distance")
        .rename("beda")
    )
    bedB = (
        dataset.ground_irradiance.isel({"distance": bedB_slice})
        .mean("distance")
        .rename("bedb")
    )
    bedC = (
        dataset.ground_irradiance.isel({"distance": bedC_slice})
        .mean("distance")
        .rename("bedc")
    )
    bedD = (
        dataset.ground_irradiance.isel({"distance": bedD_slice})
        .mean("distance")
        .rename("bedd")
    )
    bedE = (
        dataset.ground_irradiance.isel({"distance": bedE_slice})
        .mean("distance")
        .rename("bede")
    )
    bedF = (
        dataset.ground_irradiance.isel({"distance": bedF_slice})
        .mean("distance")
        .rename("bedf")
    )

    edgetoedge = (
        dataset.ground_irradiance.isel({"distance": edgetoedge_slice})
        .mean("distance")
        .rename("edgetoedge")
    )

    beds_ds = xr.merge([underpanel, edgetoedge, bedA, bedB, bedC, bedD, bedE, bedF])

    return beds_ds


def fixed_tilt_vertical_6_beds(dataset: xr.Dataset) -> xr.Dataset:
    """
    vertical fixed tilt system with 6 beds.

    # SCENARIO 10 #

    simply reuses the implementation for the tracking 6 beds because of symmetry.

    for tracking systems, measurement starts at the center of the collector and ends at the center of the collector.
    for fixed systems,    measurement starts at the left side of the left collector and ends on the left side of the next collector.

    for vertical fixed tilt systems the left side of the collector is in the same location (projected on the ground)
    so we can reuse the math for the tracking system ONLY FOR THIS SCENARIO.

    Image to be included...
    """

    return tracking_6_beds(dataset=dataset)


def iter_beds(scenario_dataset: xr.Dataset) -> xr.Dataset:
    """
    Iterate over scenario_dataset and create a beds aggregation.

    Parameters
    ----------
    scenario_dataset: xr.Dataset

    Returns
    ----------
    Resulting beds aggregate dataset contains the following datavars
        under_panel (gid, time)
        beda (gid, time)
        bedb (gid, time)
        bedc (gid, time)
        edgetoedge (gid, time)
    """

    bins = np.array([3.8, 3.8492, 4.9491, 6.9355, 11.5465, 24.0001, np.inf])

    distance_index = np.arange(0, 10)

    slices = np.array(
        [
            [0, 5, 5, 7, 7, 8, 8, 10],  # 3.8, 3.8492
            [0, 4, 4, 6, 6, 8, 8, 10],  # 3.8492,  4.9491
            [0, 3, 3, 5, 5, 8, 8, 10],  # 4.9491, 6.9355
            [0, 2, 2, 5, 5, 7, 7, 10],  # 6.9355, 11.5465
            [0, 1, 1, 4, 4, 7, 7, 10],  # 11.5465, 24.0
        ],
        dtype=int,
    )

    digitized = np.digitize(scenario_dataset.pitch, bins)

    # if anything in the range
    if np.any(digitized >= 6):
        raise ValueError("Recieved invalid pitch value above 24.")

    slice_index = digitized - 1
    selected_slices = slices[slice_index]

    res = []
    for i, gid in enumerate(scenario_dataset.gid):
        # calculate mask
        under_start = selected_slices[i, 0]
        under_end   = selected_slices[i, 1]
        beda_start  = selected_slices[i, 2]
        beda_end    = selected_slices[i, 3]
        bedb_start  = selected_slices[i, 4]
        bedb_end    = selected_slices[i, 5]  
        bedc_start  = selected_slices[i, 6]
        bedc_end    = selected_slices[i, 7]  

        mask_under = (distance_index >= under_start) & (distance_index < under_end)
        mask_edgetoedge = (distance_index >= beda_start) & (distance_index < bedc_end)
        mask_beda = (distance_index >= beda_start) & (distance_index < beda_end)
        mask_bedb = (distance_index >= bedb_start) & (distance_index < bedb_end)
        mask_bedc = (distance_index >= bedc_start) & (distance_index < bedc_end)

        single_loc_beds_ds = xr.merge(
            [
                scenario_dataset.ground_irradiance.isel(gid=i, distance=mask_under)
                .mean(dim="distance")
                .rename("under_panel"),
                scenario_dataset.ground_irradiance.isel(gid=i, distance=mask_beda)
                .mean(dim="distance")
                .rename("beda"),
                scenario_dataset.ground_irradiance.isel(gid=i, distance=mask_bedb)
                .mean(dim="distance")
                .rename("bedb"),
                scenario_dataset.ground_irradiance.isel(gid=i, distance=mask_bedc)
                .mean(dim="distance")
                .rename("bedc"),
                scenario_dataset.ground_irradiance.isel(gid=i, distance=mask_edgetoedge)
                .mean(dim="distance")
                .rename("edgetoedge"),
            ]
        )

        res.append(single_loc_beds_ds)

    return xr.concat(res, dim="gid")


def fixed_tilt_3_beds(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    fixed tilt system (latitude tilt) with optimal pitch considerations with 3 beds

    # SCENARIO 6, 7, 8, 9 #

    dataset must contain a pitch datavariable.
    """

    if "pitch" not in dataset.data_vars:
        raise ValueError("pitch data variable missing from provided dataset")

    BED_DIMS = ("gid", "time")
    for d in BED_DIMS:
        if d not in dataset.dims:
            raise ValueError(
                f"Missing required dim {d} in dataset, found {list(dataset.dims)}"
            )

    if dataset.chunks is None:
        raise RuntimeError("dataset is unchunked when it should be chunked")

    chunks = dataset.chunks
    sizes = dataset.sizes

    keys_to_take = ["gid", "time"]
    template_chunks = {
        key: value for key, value in chunks.items() if key in keys_to_take
    }

    size_gid = sizes["gid"]
    size_time = sizes["time"]

    def _empty_like_chunks():
        return da.empty(
            dtype=float,
            shape=(size_gid, size_time),
        )

    map_template = xr.Dataset(
        data_vars={
            "under_panel": (BED_DIMS, _empty_like_chunks()),
            "beda": (BED_DIMS, _empty_like_chunks()),
            "bedb": (BED_DIMS, _empty_like_chunks()),
            "bedc": (BED_DIMS, _empty_like_chunks()),
            "edgetoedge": (BED_DIMS, _empty_like_chunks()),
        },
        coords={
            "gid": dataset.gid,
            "time": dataset.time,
        },
    ).chunk(template_chunks)

    beds_ds = dataset.map_blocks(func=iter_beds, template=map_template)
    return beds_ds


def normalize_per_land_area_fixed_pitch(scenario: str, dataset: xr.Dataset):
    GROUND_AREA_OCCUPIED_ACRES_MAP = {
        "01": 0.261,
        "02": 0.261,
        "03": 0.522,
        "04": 0.417,
        "05": 0.579,
        "10": 0.453,
    }

    if scenario not in GROUND_AREA_OCCUPIED_ACRES_MAP:
        raise ValueError(
            f"invalid scenarios provided, only 01, 02, 03, 04, 05, 10 allowed, you provided {scenario}"
        )

    ground_area_occupied_acres = GROUND_AREA_OCCUPIED_ACRES_MAP[scenario]

    res = []
    for var_name in ENERGY_VARIABLES:
        res.append(
            (dataset[var_name] / ground_area_occupied_acres).rename(
                f"{var_name}_per_acre"
            )
        )

    return xr.merge(res)


def normalize_per_land_area_variable_pitch(dataset: xr.Dataset):
    TOTAL_MODULE_AREA = 0.104  # acres
    CW = 2  # meters

    # total ground area occupied by array (acres)
    ground_area_occupied_acres = TOTAL_MODULE_AREA * dataset.pitch / CW

    res = []
    for var_name in ENERGY_VARIABLES:
        res.append(
            (dataset[var_name] / ground_area_occupied_acres).rename(
                f"{var_name}_per_acre"
            )
        )

    return xr.merge(res)


def normalize_per_kWdc_installed(dataset: xr.Dataset):
    # same for all setups
    INSTALLED_CAPACITY = 80.066  # kWdc

    res = []
    for var_name in ENERGY_VARIABLES:
        res.append(
            (dataset[var_name] / INSTALLED_CAPACITY).rename(
                f"{var_name}_per_kWdc_installed"
            )
        )

    return xr.merge(res)


def farmable_land_percent(ds: xr.Dataset) -> xr.DataArray:
    pitch = ds.pitch
    CW = 2
    tracking_check = (ds.tilt == -999).all().compute().item()
    if tracking_check is True:
        logger.info("tracking configuration used")
        tilt = 0
    else:
        logger.info("using fixed tilt configuration")
        tilt = ds.tilt

    vertical_check = (ds.tilt == 90).all().compute().item()
    if vertical_check is True:
        logger.info("using farming buffer of 0.5 m for vertical configuration")
        buffer = 0.5  # 0.5 m
    else:
        logger.info(
            "using farming buffer of 0.1524 m (0.5 ft) non-vertical configuration"
        )
        buffer = 0.1524  # [m] = 0.5 ft

    xp = CW * np.cos(np.deg2rad(tilt))
    e2e = pitch - xp
    arable_land = e2e - (2 * buffer)

    farmable_percentage = (arable_land / pitch) * 100

    return farmable_percentage.rename("farmable_land_percent")


def photosynthetically_active_radiation(edge_to_edge_ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate PAR (mol m^-1 s^-1) from ground irradiance (w/m^2).
    """

    if "edgetoedge" not in edge_to_edge_ds.data_vars:
        print(edge_to_edge_ds.data_vars)
        raise ValueError("'edgetoedge' must be a datavariable in the provided dataset")

    edgetoedge_par = 2.45744 * edge_to_edge_ds.edgetoedge + 22.4778
    edgetoedge_par = edgetoedge_par.rename("edgetoedge_par")

    return edgetoedge_par

def ground_irradiance_distances(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate beds distances from pitch.
    """
    NUM_BEDS = 10
    pitch = ds.pitch

    frac = xr.DataArray(
        (np.arange(NUM_BEDS) + 0.5) / NUM_BEDS,
        coords={"distance": np.arange(10, dtype=np.int32)},
        name=f"{NUM_BEDS}_fraction",
    )

    distances = (pitch * frac).rename("distances_m")
    return distances


BED_PROCESSORS: Mapping[str, Callable[[xr.Dataset], xr.Dataset]] = {
    **{s: tracking_3_beds for s in TRACKING_3_BEDS_SCENARIOS},
    **{s: tracking_6_beds for s in TRACKING_6_BEDS_SCENARIOS},
    **{s: fixed_tilt_vertical_6_beds for s in VERTICAL_SCENARIOS},
    **{s: fixed_tilt_3_beds for s in VARIABLE_PITCH_SCENARIOS},
}

NormKind = Literal["fixed_pitch", "variable_pitch"]
NORM_KIND: Mapping[str, NormKind] = {
    **{s: "fixed_pitch" for s in (TRACKING_SCENARIOS | VERTICAL_SCENARIOS)},
    **{s: "variable_pitch" for s in VARIABLE_PITCH_SCENARIOS},
}


def postprocessing(
    scenario: str, input_zarr_paths: list[Path], output_zarr_path: Path
) -> None:
    """
    run postprocessing on model run output zarr stores.

    Parameters
    -----------
    scenario: str
        scenario name 01-10
    input_zarr_paths:
        path to zarr store containing model outputs
    output_zarr_path:
        path to write postprocessed zarr data to
    """

    if not isinstance(scenario, str):
        raise ValueError(
            f"scenario must be a string, recived {type(scenario)}: {scenario}"
        )

    if scenario not in ALL_SCENARIOS:
        raise ValueError(
            f"Invalid scenario '{scenario}'. Must be one of: {sorted(ALL_SCENARIOS)}"
        )

    if output_zarr_path.exists():
        raise FileExistsError(
            f"output_zarr_path: {output_zarr_path} already exists, must be empty."
        )

    zarrs = [xr.open_zarr(str(path)) for path in input_zarr_paths]
    scenario_dataset = xr.concat(zarrs, dim="gid").sortby("gid")
    # scenario_dataset = xr.open_zarr(input_zarr_path)

    bed_proc = BED_PROCESSORS.get(scenario)
    if bed_proc is None:
        raise RuntimeError(
            f"No bed processor mapped for scenario '{scenario}' (internal config error)."
        )

    logger.info(
        f"Postprocessing scenario={scenario} using bed_proc={bed_proc.__name__}"
    )
    beds_dataset = bed_proc(scenario_dataset)

    kind = NORM_KIND[scenario]
    if kind == "fixed_pitch":
        area_normalized_ds = normalize_per_land_area_fixed_pitch(
            scenario, scenario_dataset
        )
    else:
        area_normalized_ds = normalize_per_land_area_variable_pitch(scenario_dataset)

    kWdc_installed_normalized_ds = normalize_per_kWdc_installed(scenario_dataset)
    farmable_land_percent_da = farmable_land_percent(scenario_dataset)
    edgetoedge_par_da = photosynthetically_active_radiation(beds_dataset)
    distances_m_da = ground_irradiance_distances(scenario_dataset)

    logger.debug("Merging outputs...")
    postprocess_result_ds: xr.Dataset = xr.merge(
        [
            beds_dataset,
            area_normalized_ds,
            kWdc_installed_normalized_ds,
            farmable_land_percent_da,
            edgetoedge_par_da,
            distances_m_da
        ],
        compat="no_conflicts",
    )

    postprocess_result_ds = postprocess_result_ds.chunk({"gid": 39, "time": -1})

    logger.info("Merging outputs...")
    postprocess_result_ds.to_zarr(
        store=output_zarr_path, consolidated=True, compute=True
    )
    logger.info("Postprocessing complete")
