import pvdeg
from pathlib import Path
from dask import delayed, is_dask_collection
from dask.delayed import Delayed
from dask.distributed import as_completed, WorkerPlugin, worker
import dask.array as da
import uuid
import xarray as xr
import pandas as pd
import time
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import os
import sys

from inspire_agrivolt import logger
from inspire_agrivolt.file_operations import check_zarr_coords_dims


WEATHER_DB = "NSRDB"
WEATHER_ARG = {
    "satellite": "Americas",
    "names": "TMY",
    "NREL_HPC": True,
    "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
}

_CACHE_NSRDB: Union[tuple[xr.Dataset, pd.DataFrame], None] = None
TMY_TIME_INDEX = pd.date_range("2001-01-01", periods=8760, freq="1h")


def _as_numpy(a):
    # identity that forces numpy arrays in each block
    return np.asarray(a)


def _force_numpy_blocks(a):
    # stays lazy: converts each dask block to a plain NumPy ndarray at compute time
    return a.map_blocks(np.asarray, dtype=a.dtype)


def _force_numpy_everywhere(ds: xr.Dataset) -> xr.Dataset:
    ds2 = ds.copy()

    # coords: pin to NumPy arrays (usually tiny; OK to materialize)
    for c in ds2.coords:
        ds2[c] = xr.DataArray(
            np.asarray(ds2[c].values),  # values -> NumPy
            dims=ds2[c].dims,
            attrs=ds2[c].attrs,
        )

    # data_vars: normalize backend per block (lazy)
    for v in ds2.data_vars:
        data = ds2[v].data
        if isinstance(data, da.Array):
            ds2[v].data = _force_numpy_blocks(data)
        else:
            ds2[v].data = np.asarray(data)
    return ds2


def _assert_dask_backed(ds: xr.Dataset):
    for k, v in ds.data_vars.items():
        if not hasattr(v.data, "chunks"):
            raise RuntimeError(
                f"Variable {k} is not dask-backed; add chunks= to open_dataset()"
            )


def _assert_xr_dataset_lazy(ds: xr.Dataset) -> None:
    not_lazy = []
    for name, data_array in ds.data_vars.items():
        if not is_dask_collection(data_array.data):
            not_lazy.append(name)
    if not_lazy:
        raise RuntimeError(f"Non-lazy variables in result: {not_lazy}")


# def get_local_weather(local_test_paths: dict) -> tuple[xr.Dataset, pd.DataFrame, int]:
#     if "weather" not in local_test_paths:
#         raise ValueError('"local_test_paths" must contain key weather with value as file path')
#     if "meta" not in local_test_paths:
#         raise ValueError('"local_test_paths" must contain key meta with value as file path')

#     # need to be tmy files
#     geo_weather = xr.open_dataset(local_test_paths["weather"])
#     geo_meta = pd.read_csv(local_test_paths["meta"], index_col=0)

#     if "wind_direction" not in geo_weather.data_vars:
#         logger.info("using placeholder wind direction, filling zeros")
#         geo_weather = geo_weather.assign(wind_direction=geo_weather["temp_air"] * 0)

#     if "albedo" not in geo_weather.data_vars:
#         logger.info("using placeholder albedo, filling 0.2 for all timesteps")
#         geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2)

#     logger.info(f"{len(geo_meta)} location entries in local data")

#     # geo_weather['time'] = pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='1h')
#     chunk_size = 10

#     return geo_weather, geo_meta, chunk_size


def _log_running_worker_name(message: str) -> None:
    """
    Return worker name if execution environment is a dask worker.
    """
    logger.debug("checking worker")
    try:
        dask_worker = worker.get_worker()
        print(dask_worker, dask_worker.name)
        logger.info(message + f"(running on worker {dask_worker.name})")
    except ValueError:
        logger.info(message + "running outside of worker")


def _load_full_nsrdb_kestrel() -> tuple[xr.Dataset, pd.DataFrame]:
    global _CACHE_NSRDB
    if _CACHE_NSRDB is not None:
        logger.info("LOADING: using previously cached NSRDB")
        return _CACHE_NSRDB

    from pvdeg.utilities import nrel_kestrel_check

    nrel_kestrel_check()

    logger.info("LOADING: weather dataset from NSRDB on kestrel")
    start = time.time()

    geo_weather, geo_meta = pvdeg.weather.get(
        WEATHER_DB, geospatial=True, **WEATHER_ARG
    )

    end = time.time()
    logger.info(f"loaded: dataset in: {end - start} s")

    _CACHE_NSRDB = (geo_weather, geo_meta)  # assigns on the worker
    return _CACHE_NSRDB


def load_weather_state(
    local_test_paths: dict,
    state: str,
) -> tuple[xr.Dataset, pd.DataFrame, int]:
    if local_test_paths is not None:
        logger.info("LOADING: weather dataset from local files")
        # geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)
        raise ValueError("this functionality has been temporarily disabled")

    else:
        geo_weather, geo_meta = _load_full_nsrdb_kestrel()

        # downselect NSRDB to requested gids
        logger.info(f"DOWNSELECTING: keeping points where 'State'== '{state}'")
        start = time.time()
        geo_meta = geo_meta[geo_meta["Country"] == "United States"]
        geo_meta = geo_meta[geo_meta["State"] == state]
        end = time.time()
        logger.info(f"DOWNSELECTING: took {end - start} seconds")

        chunk_size = 40

    geo_weather = geo_weather.sel(gid=geo_meta.index).chunk({"gid": chunk_size})

    return geo_weather, geo_meta, chunk_size


def load_weather_gids(
    local_test_paths: dict,
    gids: np.ndarray,
) -> tuple[xr.Dataset, pd.DataFrame, int]:
    if local_test_paths is not None:
        logger.info("LOADING: weather dataset from local files")
        # geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)
        raise ValueError("this functionality has been temporarily disabled")

    else:
        geo_weather, geo_meta = _load_full_nsrdb_kestrel()

    chunk_size = 40

    logger.info(f"DOWNSELECTING: keeping points from provided #{len(gids)} gids")
    start = time.time()

    geo_weather = geo_weather.sel(gid=gids).chunk({"gid": chunk_size})
    geo_meta = geo_meta.loc[gids]

    end = time.time()
    logger.info(f"DOWNSELECTING: took {end - start} seconds")

    return geo_weather, geo_meta, chunk_size


class WeatherPlugin(WorkerPlugin):
    def __init__(self, logger):
        self.logger = logger

    def setup(self, worker):
        self.worker = worker
        self.logger.info(f"PLUGIN: loading NSRDB on worker {self.worker.name}")
        _load_full_nsrdb_kestrel()


class PandasNumpyPlugin(WorkerPlugin):
    def __init__(self, logger):
        self.logger = logger

    def setup(self, worker):
        import os
        import sys
        import pandas as pd

        # prove we’re on the worker
        worker.log_event(
            "plugin",
            {
                "plugin": "pandas-numpy",
                "worker": worker.name,
                "pid": os.getpid(),
                "py": sys.version,
            },
        )
        self.logger.info(
            f"[pandas-numpy] worker={worker.name} pid={os.getpid()} py={sys.version}"
        )

        # prefer numpy-backed dtypes
        try:
            pd.set_option("mode.dtype_backend", "numpy_nullable")
        except Exception as e:
            self.logger.warning(f"[pandas-numpy] failed set_option dtype_backend: {e}")

        try:
            pd.set_option("mode.string_storage", "python")
        except Exception as e:
            self.logger.warning(f"[pandas-numpy] failed set_option string_storage: {e}")

        # remove env flags that force pyarrow
        for var in ("PANDAS_USE_PYARROW", "PANDAS_STRING_STORAGE"):
            if os.environ.pop(var, None) is not None:
                self.logger.info(f"[pandas-numpy] unset {var}")

        # marker for verification
        setattr(worker, "_pandas_numpy_set", True)


def _fail_if_nans(n):
    n = int(n)
    if n > 0:
        raise RuntimeError(f"QC FAIL: {n} NaNs")
    return n


def assert_zarr_safe(ds: xr.Dataset) -> None:
    bad = []
    for name, v in {**ds.data_vars, **ds.coords}.items():
        # flag obvious problems
        if v.dtype == object:
            bad.append((name, "object dtype"))
        # try to peek at the array backend
        try:
            backend = type(getattr(v.data, "_meta", v.data)).__name__
        except Exception:
            backend = type(v.data).__name__
        if "arrow" in backend.lower():
            bad.append((name, f"backend={backend}"))
    if bad:
        lines = ", ".join([f"{n} ({why})" for n, why in bad])
        raise TypeError(f"Zarr-unsafe variables/coords: {lines}")


def _get_versions():
    """
    Get a dict of key package versions.
    """
    from importlib.metadata import version

    packages = [
        "dask",
        "dask_jobqueue",
        "numpy",
        "pandas",
        "pvdeg",
        "scipy",
        "xarray",
        "numcodecs",
        "zarr",
        "fsspec",
        "inspire_agrivolt",
    ]

    out = {"python": sys.version}
    for package in packages:
        out[package] = version(package)

    return out


def write_slice_zarr(ds: xr.Dataset, out_dir: str, stem: str) -> tuple[delayed, str]:
    """
    Generate a delayed graph to write outputs to zarr

    The following writes, validates, and renames (publishes) temporary outputs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"{stem}.{uuid.uuid4().hex}.part.zarr"  # unique per attempt
    final = out / f"{stem}.zarr"

    # why do we say compute = False then immediately compute()
    # logger.info(f"STATUS: writing gids block to zarr at: {str(partial)}")

    ds = _force_numpy_everywhere(ds)
    ds = ds.chunk({"gid": 40, "time": -1, "distance": -1})
    assert_zarr_safe(ds)

    write = ds.to_zarr(str(partial), mode="w", compute=False)

    # logger.info(f"STATUS: write finished, renaming to {str(partial)} ")

    # triggers the write (side-effecting) as a dependency
    # -> collection of paths after write
    after_write_path = delayed(lambda _w, p: p, name=f"after-write-{stem}", pure=False)(
        write, str(partial)
    )

    def _validate(path: str) -> str:
        check_zarr_coords_dims(path)
        return path

    def _publish(src, dest) -> str:
        os.replace(src, dest)
        return dest

    # this is just a QC gate to make sure that written files are valid, function returns nothing
    # WILL THIS BE A PROBLEM FOR DASK?
    validated_path = delayed(_validate, pure=False, name=f"validate-{stem}")(
        after_write_path
    )

    publish = delayed(_publish, pure=False)(validated_path, str(final))

    return publish, str(final)


def build_slice_task(
    conf: str,
    conf_dir: str,
    target_dir: str,
    sub_gids: ArrayLike,
    local_test_paths: Union[dict[str, str], None] = None,
) -> tuple[Delayed, Delayed, str]:
    """
    Return delayed taks: [nan_count, slice_task, final_path]
    """

    logger.debug("running slice task slice task...")

    # determine if this is actually lazy
    logger.debug("loading NSDRB to run slice tasks...")
    slice_weather, slice_meta, chunk_size = load_weather_gids(
        gids=sub_gids,
        local_test_paths=local_test_paths,
    )

    _assert_dask_backed(slice_weather)
    logger.debug("loaded dataset is dask backed")

    slice_weather: xr.Dataset = pvdeg.weather.map_weather(slice_weather)
    slice_weather = slice_weather.sortby("gid")
    slice_meta = slice_meta.sort_index()

    slice_template = pvdeg.geospatial.output_template(
        ds_gids=slice_weather,
        shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
        add_dims={"distance": 10},
    )
    logger.debug("created slice output template")

    # force align index, this is not ideal because it could obscure timezone errors
    # both are in UTC and off by a multiple of years so this is fine
    slice_weather = slice_weather.assign_coords(time=("time", TMY_TIME_INDEX))
    slice_template = slice_template.assign_coords(time=("time", TMY_TIME_INDEX))

    partial_res = pvdeg.geospatial.analysis(
        weather_ds=slice_weather,
        meta_df=slice_meta,
        func=pvdeg.pysam.inspire_ground_irradiance,
        template=slice_template,
        config_files={"pv": f"{conf_dir}/{conf}/{conf}_pvsamv1.json"},
        preserve_gid_dim=True,  # keep in gids, this makes it much easier to combine all of the results
        compute=False,  # DO NOT COMPUTE, build lazy object
    )

    partial_res = _force_numpy_everywhere(partial_res)

    assert_zarr_safe(partial_res)
    _assert_xr_dataset_lazy(partial_res)
    _assert_dask_backed(partial_res)

    # Warning: partial_res may not contain every GID in the range [gid_min, gid_max]
    gid_min = np.min(sub_gids)
    gid_max = np.max(sub_gids)

    out_dir = Path(target_dir) / conf
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{gid_min}-{gid_max}"

    nan_da = partial_res.to_array().isnull().sum()  # sum across all vars/dims
    nan_count = delayed(lambda x: int(x))(nan_da)

    # QC gate for model outputs, errors loudly
    qc = delayed(_fail_if_nans)(nan_count)  # may raise

    provenance = {
        "software_versions": _get_versions(),
        "kestrel_nsrdb_fnames": slice_weather.attrs.get("kestrel_nsrdb_fnames"),
    }
    partial_res = partial_res.assign_attrs(provenance)

    publish_task, final_path = write_slice_zarr(
        ds=partial_res,
        out_dir=str(out_dir),
        stem=stem,
    )

    publish_task_guarded = delayed(
        lambda _ok, task: task, pure=False, name="publish_task_qc_guarded"
    )(qc, publish_task)

    del slice_weather, slice_template, partial_res
    return nan_count, publish_task_guarded, final_path


def run_state(
    state: str,
    target_dir: Path,
    conf_dir: Path,
    dask_client,
    dask_nworkers: int,  # currently unused
    confs=[
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
    ],
    local_test_paths: dict = None,
    gids: np.ndarray = None,
    downsample: int = None,
) -> None:
    dask_client.register_plugin(
        WeatherPlugin(logger=logger), name="weather-cache-init-plugin"
    )

    # if gids are not provided then we gather them from the NSRDB to determine target gids, passed into build_slice_tasks
    if gids is None:
        init_weather, init_meta, _chunk_size = load_weather_state(
            local_test_paths=local_test_paths,
            state=state,
        )

        # downsampling is only on option if not providing pure gids
        if downsample is not None:  # optional
            logger.info(f"performing gid_downsampling() {downsample} times. ")
            init_meta, sub_gids = pvdeg.utilities.gid_downsampling(
                meta=init_meta, n=downsample
            )
            logger.info(
                f"after      gid_downsampling() {len(init_meta)} locations remain."
            )

        gids = init_meta.index.values

        logger.info(f"PREVIEW: \n{init_weather}\n")
        logger.info(f"PREVIEW: \n{init_meta.iloc[0:5]}\n")

        del init_weather, init_meta, _chunk_size
    else:
        logger.info(f"Using provided GIDS: {len(gids)}")

    logger.info(f"BUILDING: simulation on {len(gids)} unqiue locations")

    tasks = []  # delayed objects list

    # chunk size of 10, use 16 workers for this default (run as substeps so it can be handled in memory)
    # build the delayed graph
    # step_size = 640  # works but bad WHEN RUNNING 2 tasks at a time in the loop to limit parallelisim and memory usage
    step_size = 320  # 8 workers, chunk size of 40
    for conf in confs:
        logger.info(f"STATUS: building compute futures for conf: {conf}")
        # create delayed objects for each slice
        for i in range(0, len(gids), step_size):
            front, back = i, min(i + step_size, len(gids))
            sub_gids = gids[front:back]

            tasks.append(
                build_slice_task(
                    conf=conf,
                    conf_dir=str(conf_dir),
                    target_dir=str(target_dir),
                    sub_gids=sub_gids,  # only run simulation on these gids (provided, or determined at top of function)
                    local_test_paths=local_test_paths,
                )
            )

    logger.info(f"STATUS: delayed compute futures : #{len(tasks)}")
    logger.info("STATUS: dispatching dask futures...")

    max_in_flight = max(1, dask_nworkers // 2)
    ac = as_completed()

    # seed the window
    for nan_count, slice_task, final_path in tasks[:max_in_flight]:
        bundle = delayed(lambda n, _s, p: (int(n), p), pure=False)(
            nan_count, slice_task, final_path
        )
        ac.add(dask_client.compute(bundle, retries=2))

    submitted = max_in_flight
    bad = 0

    for fut in ac:
        try:
            nan_total, written_path = fut.result()
            if nan_total > 0:
                bad += 1
                logger.error(f"QC FAIL: {nan_total} NaNs -> {written_path}")
        except Exception as e:
            logger.error(f"Slice failed: {e}")
            bad += 1

        if submitted < len(tasks):
            nan_count, slice_task, final_path = tasks[submitted]
            bundle = delayed(lambda n, _s, p: (int(n), p), pure=False)(
                nan_count, slice_task, final_path
            )
            ac.add(dask_client.compute(bundle, retries=2))
            submitted += 1

    if bad:
        raise SystemExit(f"{bad} slice(s) failed QC or write.")

    return
