import argparse
import zarr
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

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
        description="Render irradiance maps for one configuration or a range of configurations."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=int,
        help="Single configuration number to render.",
    )
    parser.add_argument(
        "--dim",
        default="edgetoedge",
        help="Dataset variable to render.",
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


def single(conf_ds: xr.Dataset, i: int, dim: str) -> None:
    print("converting gids to lat lon...")
    mean_subset = pvdeg.utilities.gids_dataset_to_coords_dataset(
        conf_ds[dim].mean(dim="time"),
        gids_mapping_df
    )

    # mean_field = conf_ds[dim].mean(dim="time")
    # print("mean range:", float(mean_field.min().compute()), float(mean_field.max().compute()))
    # print("raw range :", float(conf_ds[dim].min().compute()), float(conf_ds[dim].max().compute()))
    
    # lon2d, lat2d = xr.broadcast(mean_subset.longitude, mean_subset.latitude)
    # broadcast in the order of da dims, then force exact dim order match
    lat2d, lon2d = xr.broadcast(mean_subset.latitude, mean_subset.longitude)
    lon2d = lon2d.transpose(*mean_subset.dims)
    lat2d = lat2d.transpose(*mean_subset.dims)
    
    arr = mean_subset.assign_coords(
        longitude_2d=lon2d,
        latitude_2d=lat2d,
    )

    print("plotting configuration")
    xmin, xmax = (float(arr.longitude_2d.min()), float(arr.longitude_2d.max()))
    ymin, ymax = (float(arr.latitude_2d.min()), float(arr.latitude_2d.max()))

    plot = arr.hvplot.quadmesh(
        x="longitude_2d",
        y="latitude_2d",
        z=dim,
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

    # Match the color scale to the actual field being plotted: the time-mean map
    # for this configuration, not the full underlying time series.
    vmin = float(mean_subset.min().compute())
    vmax = float(mean_subset.max().compute())

    driver = make_firefox_driver()
    plot_state = hv.renderer("bokeh").get_plot(plot).state
    finalize_bokeh_state(
        plot_state,
        title="Mean edge-to-edge irradiance [W/m²]",
        clabel="Mean edge-to-edge value",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    
    export_png(
        plot_state,
        filename=f"conf{i}-inferno-fullres.png",
        webdriver=driver,
        timeout=30,
    )
    
    driver.quit()

def render_config(config_number: int, dim: str) -> None:
    if config_number not in config_paths:
        raise ValueError(
            f"Configuration {config_number} was requested, but only "
            f"{min(config_paths)} through {max(config_paths)} are available."
        )

    print(f"running config {config_number}")
    conf_data = xr.open_zarr(config_paths[config_number])
    single(conf_ds=conf_data, i=config_number, dim=dim)


def main():
    args = parse_args()

    if args.config is not None:
        render_config(args.config, dim=args.dim)
        return

    if args.start_config > args.stop_config:
        raise ValueError("--start-config must be less than or equal to --stop-config")

    for config_number in range(args.start_config, args.stop_config + 1):
        render_config(config_number, dim=args.dim)

if __name__ == "__main__":
    main()
