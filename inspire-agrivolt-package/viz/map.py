"""
Generate publication maps


Silvana Requests
(?) cumulative GHI would be a good sanity check with the NSRDB mapts itself.
(X) Average yearly ground irradiance for setup 1 (edge to edge) 
( ) Shading factor for month of July for setup 1
( ) Ground irradiance edge to edge setup 1 for 3 pm July 21st. (I suspect this one is going to show the effect of the timezones we saw initially)

Kate Requests
( ) annual PV production/acre
( ) % farmable land per acre (variable for fixed tilt only)
"""

import argparse
import zarr
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

import dask.array as da
import dask.dataframe as dd

import datashader as ds
import colorcet as cc

import os
import sys
import holoviews as hv
import pvdeg

from bokeh.models import ColorBar, LinearColorMapper
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bokeh.io import export_png, export_svgs

import hvplot.xarray  # noqa
import cartopy.crs as ccrs
import numpy as np


def mpl_cmap_to_palette(name: str, n: int = 256) -> list[str]:
    cmap = plt.get_cmap(name)
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        )
        for r, g, b, _ in cmap(np.linspace(0, 1, n))
    ]


def finalize_bokeh_state(
    plot_state,
    *,
    title: str,
    clabel: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    # Attach title on the final exported Bokeh figure so it is not lost when
    # the upstream HoloViews object is a composite geo overlay.
    plot_state.title.text = title
    plot_state.title.text_font_size = "22pt"

    # Add a standalone Bokeh colorbar directly to the exported figure. This
    # avoids the HoloViews composite/overlay path where colorbar options may be
    # dropped or routed to the wrong sub-element.
    color_mapper = LinearColorMapper(
        palette=mpl_cmap_to_palette(cmap),
        low=vmin,
        high=vmax,
    )
    color_bar = ColorBar(
        color_mapper=color_mapper,
        title=clabel,
        width=35,
        title_standoff=10,
        major_label_text_font_size="18pt",
        title_text_font_size="20pt",
    )
    plot_state.add_layout(color_bar, "right")
    plot_state.min_border_right = max(plot_state.min_border_right, 90)


def make_firefox_driver():
    env_bin = os.path.join(sys.prefix, "bin")
    firefox_bin = os.path.join(env_bin, "firefox")
    gecko_bin = os.path.join(env_bin, "geckodriver")

    os.environ["PATH"] = env_bin + os.pathsep + os.environ["PATH"]
    os.environ["TMPDIR"] = os.path.expanduser("~/tmp")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    opts = Options()
    opts.binary_location = firefox_bin
    opts.add_argument("-headless")

    service = Service(
        executable_path=gecko_bin,
        log_output=os.path.expanduser("~/geckodriver.log"),
    )

    return webdriver.Firefox(service=service, options=opts)

live_data_copy = Path("/projects/inspire/PySAM-MAPS/v1.2/")
data_dir = live_data_copy / "final-backup/"
gid_file = live_data_copy / "gid-lat-lon.csv"

gids_mapping_df = pd.read_csv(gid_file, index_col=0)
gids_mapping_df.index.name = "gid"

confs = sorted(list(data_dir.iterdir()))
config_paths = {
    config_number: conf_path
    for config_number, conf_path in enumerate(confs, start=1)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render irradiance maps for the built-in visualization modes."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "mean-edgetoedge",
            "july-shading-factor-setup1",
            "july-21-15-edgetoedge-setup1",
        ),
        default="mean-edgetoedge",
        help=(
            "Field selection to render. "
            "'mean-edgetoedge' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--start-config",
        type=int,
        default=1,
        help="First configuration number to render, inclusive.",
    )
    parser.add_argument(
        "--stop-config",
        type=int,
        default=11,
        help="Last configuration number to render, inclusive.",
    )
    return parser.parse_args()


def build_fallback_hourly_index(size: int) -> pd.DatetimeIndex:
    if size != 8760:
        raise ValueError(
            "Expected 8760 hourly steps when reconstructing a fallback time index, "
            f"but received {size}."
        )
    return pd.date_range("2001-01-01 00:00:00", periods=size, freq="1h")


def get_hourly_index(conf_ds: xr.Dataset) -> pd.DatetimeIndex:
    if "time" not in conf_ds.coords:
        raise ValueError("Dataset is missing a 'time' coordinate.")

    time_values = conf_ds["time"].values
    if is_datetime64_any_dtype(time_values):
        return pd.DatetimeIndex(time_values)

    print(
        "time coordinate is not datetime-like; assuming local hourly indexing "
        "from 2001-01-01 00:00:00 through 2001-12-31 23:00:00"
    )
    return build_fallback_hourly_index(conf_ds.sizes["time"])


def get_time_mask(conf_ds: xr.Dataset, *, month: int | None = None) -> xr.DataArray:
    hourly_index = get_hourly_index(conf_ds)
    mask = pd.Series(True, index=hourly_index)

    if month is not None:
        mask &= hourly_index.month == month

    return xr.DataArray(mask.to_numpy(), dims=("time",), coords={"time": conf_ds["time"]})


def get_time_position(conf_ds: xr.Dataset, timestamp: str) -> int:
    hourly_index = get_hourly_index(conf_ds)
    target_time = pd.Timestamp(timestamp)

    try:
        return hourly_index.get_loc(target_time)
    except KeyError as exc:
        raise ValueError(f"{target_time} is not available in the dataset time index.") from exc


def mean_edgetoedge_over_time(conf_ds: xr.Dataset) -> xr.DataArray:
    return conf_ds["edgetoedge"].mean(dim="time").rename("edgetoedge")


def july_shading_factor_setup1(conf_ds: xr.Dataset) -> xr.DataArray:
    july_mask = get_time_mask(conf_ds, month=7)
    july_edgetoedge = conf_ds["edgetoedge"].where(july_mask, drop=True).mean(dim="time")
    july_ghi = conf_ds["ghi"].where(july_mask, drop=True).mean(dim="time")
    shading_factor = (july_edgetoedge / july_ghi).rename("shading_factor")
    return shading_factor


def july_21_15_edgetoedge_setup1(conf_ds: xr.Dataset) -> xr.DataArray:
    time_position = get_time_position(conf_ds, "2001-07-21 15:00:00")
    return conf_ds["edgetoedge"].isel(time=time_position).rename("edgetoedge")


def select_field(conf_ds: xr.Dataset, *, config_number: int, mode: str) -> xr.DataArray:
    if mode == "mean-edgetoedge":
        return mean_edgetoedge_over_time(conf_ds)

    if config_number != 1:
        raise ValueError(f"{mode} is only defined for setup/configuration 1.")

    if mode == "july-shading-factor-setup1":
        return july_shading_factor_setup1(conf_ds)

    if mode == "july-21-15-edgetoedge-setup1":
        return july_21_15_edgetoedge_setup1(conf_ds)

    raise ValueError(f"Unsupported render mode: {mode}")


def get_plot_metadata(mode: str, field_name: str) -> dict[str, str]:
    if mode == "mean-edgetoedge":
        return {
            "title": "Mean edge-to-edge irradiance over time",
            "clabel": "Mean edge-to-edge irradiance [W/m²]",
        }

    if mode == "july-shading-factor-setup1":
        return {
            "title": "July shading factor for setup 1",
            "clabel": "Shading factor",
        }

    if mode == "july-21-15-edgetoedge-setup1":
        return {
            "title": "Setup 1 edge-to-edge irradiance on July 21 at 3 PM",
            "clabel": "Edge-to-edge irradiance [W/m²]",
        }

    return {
        "title": field_name,
        "clabel": field_name,
    }


def get_output_stem(mode: str) -> str:
    if mode == "mean-edgetoedge":
        return "inferno-fullres"
    return f"{mode}-inferno-fullres"


def single(conf_ds: xr.Dataset, i: int, mode: str) -> None:
    print("converting gids to lat lon...")
    selected_field = select_field(
        conf_ds,
        config_number=i,
        mode=mode,
    )
    subset = pvdeg.utilities.gids_dataset_to_coords_dataset(selected_field, gids_mapping_df)

    # broadcast in the order of da dims, then force exact dim order match
    lat2d, lon2d = xr.broadcast(subset.latitude, subset.longitude)
    lon2d = lon2d.transpose(*subset.dims)
    lat2d = lat2d.transpose(*subset.dims)

    arr = subset.assign_coords(
        longitude_2d=lon2d,
        latitude_2d=lat2d,
    )

    print("plotting configuration")
    xmin, xmax = (float(arr.longitude_2d.min()), float(arr.longitude_2d.max()))
    ymin, ymax = (float(arr.latitude_2d.min()), float(arr.latitude_2d.max()))

    plot = arr.hvplot.quadmesh(
        x="longitude_2d",
        y="latitude_2d",
        z=selected_field.name,
        geo=True,
        crs=ccrs.PlateCarree(),
        projection=ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=37.5,
            standard_parallels=(29.5, 45.5),
        ),
        project=True,
        rasterize=True,
        coastline=True,
        features=['states', 'borders'],
        cmap="inferno",
        width=4000,
        height=2400,
        # width=400,
        # height=240,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )

    vmin = float(subset.min().compute())
    vmax = float(subset.max().compute())
    metadata = get_plot_metadata(mode, selected_field.name)

    driver = make_firefox_driver()
    plot_state = hv.renderer("bokeh").get_plot(plot).state
    finalize_bokeh_state(
        plot_state,
        title=metadata["title"],
        clabel=metadata["clabel"],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    
    export_png(
        plot_state,
        filename=f"conf{i}-{get_output_stem(mode)}.png",
        webdriver=driver,
        timeout=30,
    )
    
    driver.quit()

def render_config(config_number: int, mode: str) -> None:
    if config_number not in config_paths:
        raise ValueError(
            f"Configuration {config_number} was requested, but only "
            f"{min(config_paths)} through {max(config_paths)} are available."
        )

    print(f"running config {config_number}")
    conf_data = xr.open_zarr(config_paths[config_number])
    single(conf_ds=conf_data, i=config_number, mode=mode)


def main():
    args = parse_args()

    if args.mode == "mean-edgetoedge":
        for config_number in config_paths:
            render_config(config_number, mode=args.mode)
        return

    render_config(1, mode=args.mode)

if __name__ == "__main__":
    main()
