import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from dask.distributed import LocalCluster, Client

### TAKEN FROM beds_postprocessing
def ground_irradiance_distances(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate beds distances from pitch.

    TAKEN FROM beds_postprocessing
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("final_dir", type=str, help="inspire final directory")

    args = parser.parse_args()
    final_dir = args.final_dir
    print(f"using final_dir: {final_dir}")


    with LocalCluster(n_workers=16) as cluster, Client(cluster) as client:
        print(client.dashboard_link)

        for item in Path(final_dir).iterdir():
            if not item.is_dir() or item.suffix != ".zarr":
                continue

            try:
                print(f"trying {str(item)}")
                ds = xr.open_zarr(item)
                distances_m_da = ground_irradiance_distances(ds)

                # mirror notebook behavior (DataArray -> to_zarr append)
                write = distances_m_da.to_zarr(store=item, mode="a")
                write.compute()

            except Exception as e:
                print(f"{item} failed ({type(e).__name__}: {e})")
                continue
            except FileNotFoundError:
                print(str(item), "could not open zarr, skipping")


            loaded = xr.open_zarr(item)
            if "distances_m" in loaded.data_vars:
                print("found distances_m in zarr", item)
        
                dm = loaded.distances_m

                # false means there are only finite values in the dataset
                nonfinite = (~xr.apply_ufunc(np.isfinite, dm, dask='allowed')).any().compute().item()
                dm_min = dm.min().compute().item()
                dm_max = dm.max().compute().item()
                
                print(item, "nonfinite?", nonfinite, "min", dm_min, "max", dm_max)

if __name__ == "__main__":
    main()