import inspire_agrivolt
from dask.distributed import LocalCluster
from pathlib import Path

import argparse


CONFS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
VERSION = "v1.2"

STATES = [
    "alabama",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new_hampshire",
    "new_jersey",
    "new_mexico",
    "new_york",
    "north_carolina",
    "north_dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode_island",
    "south_carolina",
    "south_dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west_virginia",
    "wisconsin",
    "wyoming",
]


def check_all_state_confs_exist(model_outs_dir: Path):
    for state in STATES:
        for conf in CONFS:
            print(state, conf)
            path = model_outs_dir / state / conf

            if not path.exists():
                raise ValueError(f"{state}/{conf} doesn't exist")

def run_postprocess_all_states_confs(model_outs_dir: Path, postprocess_dir: Path):
    for state in STATES:
        for conf in CONFS:
            print(state, conf)
            path = model_outs_dir / state / conf

            model_outs_paths = list(
                path.glob("*.zarr")
            )

            POSTPROCESS_OUTS_ZARR = Path(
                postprocess_dir / state / conf / ".zarr"
            )

            inspire_agrivolt.beds_postprocessing.postprocessing(
                scenario=conf,
                input_zarr_paths=model_outs_paths,
                output_zarr_path=POSTPROCESS_OUTS_ZARR,
            )

def main():
    parser = argparse.ArgumentParser("postprocessor", description='postprocess model outputs')

    parser.add_argument('model_outs_dir')
    parser.add_argument('postprocess_dir')

    args = parser.parse_args()

    model_outs_dir = args.model_outs_dir
    postprocess_dir = args.postprocess_dir

    if not model_outs_dir.exists():
        raise FileNotFoundError(f"model-outputs-directory not found at {str(model_outs_dir)}")

    check_all_state_confs_exist(model_outs_dir=model_outs_dir)

    cluster = LocalCluster(
        n_workers=16, 
        threads_per_worker=1
    )
    client = cluster.get_client()

    print(client.dashboard_link)
    run_postprocess_all_states_confs(
        model_outs_dir=model_outs_dir,
        postprocess_dir=postprocess_dir
    )

if __name__ == "__main__":
    main()