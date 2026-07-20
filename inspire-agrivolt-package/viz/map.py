"""
Generate publication maps

Set up on Kestrel
----------------

module load conda
source activate /projects/inspire/envs/render/
export SSL_CERT_FILE="$CONDA_PREFIX/ssl/cert.pem"
export REQUESTS_CA_BUNDLE="$CONDA_PREFIX/ssl/cert.pem"

Iteration 2
------------

( ) july shading factor conf 1
( ) july shading factor conf 6
( ) july shading factor conf 11

( ) [uniform cbar range] july shading factor conf 1
( ) [uniform cbar range] july shading factor conf 6
( ) [uniform cbar range] july shading factor conf 11

( ) Annual insolation conf 1
( ) Annual insolation conf 6
( ) Annual insolation conf 11

( ) [uniform cbar range] annual insolation conf 1
( ) [uniform cbar range] annual insolation conf 6
( ) [uniform cbar range] annual insolation conf 11

*higher res borders*
*great lakes coloring removed* => photoshop this out


Iteration 1
-----------
Silvana Requests
(?) cumulative GHI would be a good sanity check with the NSRDB mapts itself.
(X) Average yearly ground irradiance for setup 1 (edge to edge) 
(X) Shading factor for month of July for setup 1
(X) Ground irradiance edge to edge setup 1 for 3 pm July 21st. (I suspect this one is going to show the effect of the timezones we saw initially)

Kate Requests
( ) annual PV production/acre
(X) % farmable land per acre (variable for fixed tilt only)
"""

import argparse
import calendar
import warnings
import zarr
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from zoneinfo import ZoneInfo

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
from name_map import (
    convert_kestrel_name_to_published_name,
    convert_published_name_to_kestrel_name,
)

RENDER_MODES = (
    "annual-energy-per-acre",
    "mean-daily-insolation",
    "mean-edgetoedge",
    "farmable-land-percent",
    "shading-factor",
    "july-shading-factor",
    "july-21-15-edgetoedge",
)

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

try:
    from timezonefinder import TimezoneFinder
except ImportError:  # pragma: no cover - optional dependency on the cluster
    TimezoneFinder = None


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
    firefox_bin = os.path.join(env_bin, "FirefoxApp", "firefox")
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
        choices=RENDER_MODES,
        default="mean-edgetoedge",
        help=(
            "Field selection to render. "
            "'mean-edgetoedge' preserves the current behavior. "
            "'july-shading-factor' is an alias for shading-factor with month=7."
        ),
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        default=None,
        metavar="N",
        help=(
            "Optional month (1-12) for mean-daily-insolation and shading-factor. "
            "Omit for annual / full-year aggregation."
        ),
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        required=True,
        help="Configuration numbers to render.",
    )
    parser.add_argument(
        "--config-names",
        choices=("kestrel-names", "publication-names"),
        required=True,
        help=(
            "Whether --configs are the original Kestrel/on-disk names or the "
            "publication names."
        ),
    )
    parser.add_argument(
        "--cmin",
        type=float,
        default=None,
        help="Optional lower bound for the color scale.",
    )
    parser.add_argument(
        "--cmax",
        type=float,
        default=None,
        help="Optional upper bound for the color scale.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for map PNG outputs (defaults to cwd).",
    )
    return parser.parse_args()


def normalize_mode_and_month(mode: str, month: int | None) -> tuple[str, int | None]:
    if mode == "july-shading-factor":
        if month is None:
            month = 7
        return "shading-factor", month
    return mode, month


def resolve_config_names(
    requested_configs: list[int],
    *,
    config_names: str,
) -> list[tuple[int, int]]:
    if config_names == "publication-names":
        warnings.warn(
            "Interpreting --configs as publication names and mapping them to "
            "Kestrel/on-disk names before loading zarr stores.",
            stacklevel=2,
        )
        return [
            (convert_published_name_to_kestrel_name(config_number), config_number)
            for config_number in requested_configs
        ]

    warnings.warn(
        "Interpreting --configs as Kestrel/on-disk names. Output filenames will "
        "use the corresponding publication names, which may differ. See the "
        "inspire-agrivolt-package README.md deployment section for details.",
        stacklevel=2,
    )
    return [
        (config_number, convert_kestrel_name_to_published_name(config_number))
        for config_number in requested_configs
    ]


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


def days_in_period(conf_ds: xr.Dataset, *, month: int | None) -> int:
    hourly_index = get_hourly_index(conf_ds)
    if month is None:
        return 365

    years = hourly_index[hourly_index.month == month].year
    if len(years) == 0:
        raise ValueError(f"No timesteps found for month={month}.")
    year = int(years[0])
    return calendar.monthrange(year, month)[1]


def shading_factor(conf_ds: xr.Dataset, *, month: int | None = None) -> xr.DataArray:
    time_mask = get_time_mask(conf_ds, month=month)
    period_edgetoedge = conf_ds["edgetoedge"].where(time_mask, drop=True).mean(dim="time")
    period_ghi = conf_ds["ghi"].where(time_mask, drop=True).mean(dim="time")
    return ((period_edgetoedge / period_ghi) * 100).rename("shading_factor")


def july_shading_factor(conf_ds: xr.Dataset) -> xr.DataArray:
    return shading_factor(conf_ds, month=7)


def get_gid_lat_lon(conf_ds: xr.Dataset) -> pd.DataFrame:
    gids = pd.Index(conf_ds["gid"].values, name="gid")
    gid_subset = gids_mapping_df.reindex(gids)

    if gid_subset[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing latitude/longitude for one or more gids in the mapping file.")

    return gid_subset[["latitude", "longitude"]]


def approximate_timezone_name(longitude: float) -> str:
    if longitude >= -82.5:
        return "America/New_York"
    if longitude >= -97.5:
        return "America/Chicago"
    if longitude >= -112.5:
        return "America/Denver"
    if longitude >= -127.5:
        return "America/Los_Angeles"
    return "America/Anchorage"


def get_timezone_names_for_gids(conf_ds: xr.Dataset) -> pd.Series:
    gid_lat_lon = get_gid_lat_lon(conf_ds)

    if TimezoneFinder is None:
        print(
            "timezonefinder is unavailable; approximating local timezones from longitude bands"
        )
        timezone_names = gid_lat_lon["longitude"].map(approximate_timezone_name)
    else:
        timezone_finder = TimezoneFinder(in_memory=True)

        def lookup_timezone(row: pd.Series) -> str:
            timezone_name = timezone_finder.timezone_at(
                lng=float(row["longitude"]),
                lat=float(row["latitude"]),
            )
            if timezone_name is None:
                timezone_name = approximate_timezone_name(float(row["longitude"]))
            return timezone_name

        timezone_names = gid_lat_lon.apply(lookup_timezone, axis=1)

    timezone_names.index = gid_lat_lon.index
    return timezone_names


def get_local_time_positions_for_gids(conf_ds: xr.Dataset, local_timestamp: str) -> xr.DataArray:
    hourly_index = get_hourly_index(conf_ds)
    timezone_names = get_timezone_names_for_gids(conf_ds)
    local_time = pd.Timestamp(local_timestamp)

    time_positions = pd.Series(index=timezone_names.index, dtype=np.int64)

    for timezone_name in timezone_names.unique():
        utc_time = local_time.tz_localize(ZoneInfo(timezone_name)).tz_convert("UTC").tz_localize(None)
        try:
            time_position = hourly_index.get_loc(utc_time)
        except KeyError as exc:
            raise ValueError(
                f"UTC time {utc_time} derived from local time {local_time} "
                f"for timezone {timezone_name} is not available in the dataset."
            ) from exc

        time_positions.loc[timezone_names == timezone_name] = int(time_position)

    return xr.DataArray(
        time_positions.to_numpy(),
        dims=("gid",),
        coords={"gid": conf_ds["gid"]},
    )

def mean_daily_insolation(
    conf_ds: xr.Dataset,
    *,
    month: int | None = None,
) -> xr.DataArray:
    days = days_in_period(conf_ds, month=month)
    if month is None:
        values = conf_ds["edgetoedge"].sum("time") / days / 1000
    else:
        time_mask = get_time_mask(conf_ds, month=month)
        values = (
            conf_ds["edgetoedge"].where(time_mask, drop=True).sum("time") / days / 1000
        )
    return values.rename("mean_daily_insolation")


def annual_energy_per_acre(conf_ds: xr.Dataset) -> xr.DataArray:
    # Zarr stores kWh/year/acre; convert to MWh/year/acre for maps/colorbars.
    return (conf_ds["annual_energy_per_acre"] / 1000.0).rename("annual_energy_per_acre")


def july_21_15_edgetoedge_setup(conf_ds: xr.Dataset) -> xr.DataArray:
    time_positions = get_local_time_positions_for_gids(conf_ds, "2001-07-21 15:00:00")
    return conf_ds["edgetoedge"].isel(time=time_positions).rename("edgetoedge")


def farmable_land_percent_per_acre(conf_ds: xr.Dataset) -> xr.DataArray:
    return conf_ds["farmable_land_percent"].rename("farmable_land_percent")


def select_field(
    conf_ds: xr.Dataset,
    *,
    mode: str,
    month: int | None = None,
) -> xr.DataArray:
    mode, month = normalize_mode_and_month(mode, month)

    if mode == "annual-energy-per-acre":
        return annual_energy_per_acre(conf_ds)

    if mode == "mean-daily-insolation":
        return mean_daily_insolation(conf_ds, month=month)

    if mode == "mean-edgetoedge":
        return mean_edgetoedge_over_time(conf_ds)

    if mode == "farmable-land-percent":
        return farmable_land_percent_per_acre(conf_ds)

    if mode == "shading-factor":
        return shading_factor(conf_ds, month=month)

    if mode == "july-21-15-edgetoedge":
        return july_21_15_edgetoedge_setup(conf_ds)

    raise ValueError(f"Unsupported render mode: {mode}")


def period_title_prefix(month: int | None) -> str:
    if month is None:
        return "Annual"
    return MONTH_NAMES[month]


def get_plot_metadata(
    mode: str,
    field_name: str,
    *,
    month: int | None = None,
) -> dict[str, str]:
    mode, month = normalize_mode_and_month(mode, month)

    if mode == "annual-energy-per-acre":
        return {
            "title": "Annual PV Production per Acre",
            "clabel": "Annual PV Production per Acre (MWh/year/acre)",
        }

    if mode == "mean-daily-insolation":
        return {
            "title": f"{period_title_prefix(month)} Mean Daily Insolation",
            "clabel": "Mean Daily Insolation (kWh/m²/day)",
        }

    if mode == "mean-edgetoedge":
        return {
            "title": "Mean Edge-to-Edge Irradiance Over Time",
            "clabel": "Mean Edge-to-Edge Irradiance (W/m²)",
        }

    if mode == "farmable-land-percent":
        return {
            "title": "Farmable Land Percent",
            "clabel": "Farmable Land Percent (%)",
        }

    if mode == "shading-factor":
        return {
            "title": f"{period_title_prefix(month)} Shading Factor",
            "clabel": "Shading Factor (%)",
        }

    if mode == "july-21-15-edgetoedge":
        return {
            "title": "Edge-to-Edge Irradiance on July 21 at 3 PM",
            "clabel": "Edge-to-Edge Irradiance (W/m²)",
        }

    return {
        "title": field_name,
        "clabel": field_name,
    }


def get_output_stem(mode: str, *, month: int | None = None) -> str:
    mode, month = normalize_mode_and_month(mode, month)

    if mode == "mean-edgetoedge":
        return "inferno-fullres"

    if mode in {"mean-daily-insolation", "shading-factor"}:
        if month is None:
            period = "annual"
        else:
            period = f"m{month:02d}"
        return f"{mode}-{period}-inferno-fullres"

    return f"{mode}-inferno-fullres"


def get_output_filename(
    publication_config_number: int,
    mode: str,
    *,
    month: int | None = None,
    fixed_crange: bool = False,
) -> str:
    suffix = "-fixed-crange" if fixed_crange else ""
    return (
        f"conf{publication_config_number}-"
        f"{get_output_stem(mode, month=month)}{suffix}.png"
    )


def single(
    conf_ds: xr.Dataset,
    *,
    kestrel_config_number: int,
    publication_config_number: int,
    mode: str,
    month: int | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    output_dir: Path | None = None,
) -> None:
    mode, month = normalize_mode_and_month(mode, month)

    print("converting gids to lat lon...")
    selected_field = select_field(
        conf_ds,
        mode=mode,
        month=month,
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

    clim = None
    if cmin is not None or cmax is not None:
        if cmin is None or cmax is None:
            raise ValueError("Provide both --cmin and --cmax, or neither.")
        if cmin > cmax:
            raise ValueError("--cmin must be less than or equal to --cmax.")
        clim = (cmin, cmax)

    quadmesh_kwargs = dict(
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
        # coastline=True,
        # features=['states', 'borders'],
        features={
            'states':'10m',
            'borders':'10m',
        },
        cmap="inferno",
        width=4000,
        height=2400,
        # width=400,
        # height=240,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )
    if clim is not None:
        quadmesh_kwargs["clim"] = clim

    plot = arr.hvplot.quadmesh(**quadmesh_kwargs)

    if clim is None:
        vmin = float(subset.min().compute())
        vmax = float(subset.max().compute())
    else:
        vmin, vmax = clim

    metadata = get_plot_metadata(mode, selected_field.name, month=month)

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

    fname = get_output_filename(
        publication_config_number,
        mode,
        month=month,
        fixed_crange=clim is not None,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = str(output_dir / fname)

    export_png(
        plot_state,
        filename=fname,
        webdriver=driver,
        timeout=30,
    )
    print("saved to", fname)

    driver.quit()


def render_config(
    *,
    kestrel_config_number: int,
    publication_config_number: int,
    mode: str,
    month: int | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    output_dir: Path | None = None,
) -> None:
    if kestrel_config_number not in config_paths:
        raise ValueError(
            f"Kestrel configuration {kestrel_config_number} was requested, but only "
            f"{min(config_paths)} through {max(config_paths)} are available."
        )

    print(
        f"running publication config {publication_config_number} "
        f"from Kestrel config {kestrel_config_number}"
    )
    conf_data = xr.open_zarr(config_paths[kestrel_config_number])
    single(
        conf_ds=conf_data,
        kestrel_config_number=kestrel_config_number,
        publication_config_number=publication_config_number,
        mode=mode,
        month=month,
        cmin=cmin,
        cmax=cmax,
        output_dir=output_dir,
    )


def main():
    args = parse_args()
    mode, month = normalize_mode_and_month(args.mode, args.month)

    if month is not None and mode not in {"mean-daily-insolation", "shading-factor"}:
        raise ValueError(
            f"--month is only supported for mean-daily-insolation and shading-factor, "
            f"not {mode}."
        )

    requested_configs = resolve_config_names(
        args.configs,
        config_names=args.config_names,
    )

    for kestrel_config_number, publication_config_number in requested_configs:
        render_config(
            kestrel_config_number=kestrel_config_number,
            publication_config_number=publication_config_number,
            mode=mode,
            month=month,
            cmin=args.cmin,
            cmax=args.cmax,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
