# InSPIRE_Agrivolt

``inspire_agrivolt`` is a collection of CLI tools and HPC scripts to assist in the creation of the 2025 InSPIRE Agrivoltaics Irradiance Dataset.

## Installing

The dataset was produced on Kestrel with the following [conda environment](https://github.com/tobin-ford/conda-envs/blob/main/working-geospatial.yml).

Clone the repo with one of the following

    git clone git@github.com:NREL/InSPIRE.git     # (SSH)
    git clone https://github.com/NREL/InSPIRE.git # (HTTPS)

Install the package from `InSPIRE/inspire-agrivolt-package/` with
    pip install -e .

Currently the package requires an editable `pvdeg` install on branch pysam.
See directions [here](https://github.com/NREL/PVDegradationTools?tab=readme-ov-file#installation)


## Workflow
This section describes how to create dataset using the scripts in `scripts/`. It consists of three main steps.

1. Model Runs  
2. Postprocessing  
3. Dataset Merging  
4. Deployment (Optional)  

**To run the scripts for the workflows you must define**.

1. INSPIRE_AGRIVOLT_CONDA_ENV  
    ```
    Set to the name or path of the conda/mamba environment, e.g.:

    export INSPIRE_AGRIVOLT_CONDA_ENV=/projects/inspire/envs/inspire-agrivolt-geospatial
    # or
    export INSPIRE_AGRIVOLT_CONDA_ENV=geospatial
    ```

2. INSPIRE_AGRIVOLT_SAM_CONFIG_DIR
    Set to the path of the SAM configuration directiory, e.g.:

        export INSPIRE_AGRIVOLT_SAM_CONFIG_DIR=/path/to/InSPIRE/Studies/USMap_Doubleday_2024/SAM

        The directory should have the following structure.
        InSPIRE/Studies/USMap_Doubleday_2024/SAM
        ├── 01
        │   ├── 01_pvsamv1.json
        ├── 02
        │   ├── 02_pvsamv1.json
        ├── 03
        │   ├── 03_pvsamv1.json
        ├── 04
        │   ├── 04_pvsamv1.json
        ├── 05
        │   ├── 05_pvsamv1.json
        ├── 06
        │   ├── 06_pvsamv1.json
        ├── 07
        │   ├── 07_pvsamv1.json
        ├── 08
        │   ├── 08_pvsamv1.json
        ├── 09
        │   ├── 09_pvsamv1.json
        ├── 10
        │   ├── 10_pvsamv1.json
        ├── 11
        │   ├── 11_pvsamv1.json
    
    **SAM config directories use the original config names rather than the publication names see the table in the deployment section for more information.**

    **DO NOT ATTEMPT TO CHANGE THESE TO MATCH THE PUBLICATION NAMES**. The pipeline handles this for us during the Deployment step.

    If you reorder configs, it will break hardcoded assumptions about what each of the configs mean within the code. If you add a new config you will need to go in and modify the code to support the new config. **DO NOT CHANGE THE existing configs' meaning in the code or configuration files' names, this will cause a nightmarish headache.** Instead, add new configs by incrementing the highest existing number and updating the available configs in the code. 
    
    This is the most fragile part of the pipeline because assume configs based on their filepaths and filenames so follow the existing convention and create new configs as the next greatest integer at `InSPIRE/Studies/USMap_Doubleday_2024/SAM/XX/XX_pvsamv1.json`.

3. INSPIRE_AGRIVOLT_MODEL_OUTS_DIR
    Set to the path of the desired agrivoltaic irradiance outputs directory, e.g:

        export INSPIRE_AGRIVOLT_MODEL_OUTS_DIR=/projects/inspire/PySAM-MAPS/v1.2/model-outs/

    This is where the outputs of the irradiance model go. 
    Produced by the following scripts in `scripts/`:
    - `submit_state_conf.sh`
    - `run_all_configs_state.sh`
    - `run_all_states_config.sh`

4. INSPIRE_AGRIVOLT_POSTPROCESS_DIR
    Set to the path of the desired postprocessing outputs directory, e.g:

        export INSPIRE_AGRIVOLT_POSTPROCESS_DIR=/projects/inspire/PySAM-MAPS/v1.2/postprocess/

    This is where the outputs of the postprocessing step go. Produced by the following script in `scripts/`:
    - `submit_postprocess.slurm`

5. INSPIRE_AGRIVOLT_FINAL_DIR
    Set to the path of the desired final combined outputs directory, e.g:

        export INSPIRE_AGRIVOLT_FINAL_DIR=/projects/inspire/PySAM-MAPS/v1.2/final/

    This is where the final outputs of the combine step are stored. This step takes results from `INSPIRE_AGRIVOLT_MODEL_OUTS_DIR` and `INSPIRE_AGRIVOLT_POSTPROCESSING_DIR` and combined them into their final state. No more processing is done on the combined files in `INSPIRE_AGRIVOLT_FINAL_DIR`. They are produced by the following script in `scripts/`:
    - `submit_combine.slurm`.

### 1. Model Runs
PySAM model runs as defined by the inspire_agrivolt cli.

Generally, we take the approach of running one slurm job for each (state, config) pair. Each slurm job is allocated 1 node and parallelized with Dask.

Use `scripts/run_all_configs_state.sh` to produce the model outputs. Or use  `scripts/submit_state_conf.sh` to submit a single pair at a time.

**Configuration details**
- Model run configrations are defined in `scripts/submit_state_conf.sh`.
    - Slurm configuration options
    - Dask scheduler options
    - OUTPUT_DIR: model output directory, 
    - CONF_DIF: SAM configuration directory with structure as described in (SAM Config Directories Structure)[SAM Config Directories Structure]

From working directory `inspire-agrivolt-package` run sbatch `scripts/submit_state_conf.sh`.

These currently write state output directories to an output directory defined in `scripts/submit_state_conf.sh`. These should be moved to a `model-outs/` directory for maintainability.

*Slurm sends emails based on job progress and error codes but these can fail to alert you if a job has failed at certain stages*. By defualt, the script will run chunks of ~200 GIDs from the NSRDB and output these to zarr files in the output directory. These outputs will be named `*.part.zarr` to indicate that these are paritally completed results. When Dask has finished handling a chunk, we will update the filename to `*.zarr`. If an output directory contains `*.part.zarr` outputs after the SLURM job has finished then something unexpected happened. This can occur due to timeout, job failure or CPU starvation. The easiest fix is as follows
1. Delete the failed state-conf pair in the outputs directory
2. (Optional) increase the amount of memory allocated to the job in `scripts/submit_state_conf.sh`
3. Rerun the state-conf pair.

This may be a useful command to see if the `*.part.zarr` files persist at the end of a job. 

    find . -mindepth 3 -maxdepth 3 -type d -name '*.part.zarr' -printf '%P\n' | cut -d/ -f1-2 | sort -u

Once the above command finds no partial zarrs in the output then we can move on to postprocessing.

### 2. Postprocessing
inspire_agrivolt implements postprocessing in the source `inspire_agrivolt/beds_postprocessing.py` and calls this functionality in `scripts/postprocess.py`. To kick off the slurm job to postprocess all state-config pairs, use `sbatch scripts/submit_postprocess.slurm` while in `inspire-agrivolt-package` as a working directory.

We grab all model outputs per state and combine them to run aggregation and normalization based on InSPIRE decided factors and useful unit conversions.

### 3. Dataset Merging
Combining model runs and postprocessing results into a final result.

To merge model runs and postprocessing use `sbatch scripts/submit_combine.slurm`. Then use `scripts/submit_check_combine.slurm` to check for data integrity. This only examines that we have no gids in the dataset.

### Check and Add distances_m to zarrs (OPTIONAL)

Due to a bug in the beds_postprocessing code. It is possible that  `distances_m` was not correctly appended to the zarrs during postprocessing (step 2). There is a small script called `scripts/restore_distances_m.py` that can be run as follows. 

```bash
python scripts/restore_distances_m.py $INSPIRE_AGRIVOLT_FINAL_DIR
```

The script updates existing zarrs by appending a new data variable that contains the physical distances in meters corresponding to the distances index (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) on combined datasets/zarrs.

**Note: Script updates zarrs in-place and checks that chunks are intact by looking for any non-finite values and outputting the min and max**

### 4. Deployment (OPTIONAL)
Uploading the final result to OpenEI on S3.

We want to deploy the final dataset versions to S3. AWS CLI is too slow so we will use `scripts/submit_s3_upload_zarrs.slurm`. 

**Note: this script renames zarr configs when they are uploaded to s3. It does not change them in-place (on kestrel)**


| Config name                                      | Original config | Updated config name |
| ------------------------------------------------ | --------------- | ------------------- |
| SAT (Conventional)                               | 01              | 01                  |
| SAT (Elevated)                                   | 02              | 02                  |
| SAT (Elevated with Inter-Panel Spacing)          | 03              | 03                  |
| SAT (Double Row Spacing)                         | 04              | 04                  |
| SAT (Triple Row Spacing)                         | 05              | 05                  |
| Fixed Tilt (Conventional, ground clearance 1.5m) | **06**          | **07**              |
| Fixed Tilt (Elevated)                            | **07**          | **08**              |
| Fixed Tilt (Elevated with Inter-Panel Spacing)   | **08**          | **09**              |
| Fixed Tilt (Elevated with Inter-Panel Spacing)   | **09**          | **10**              |
| Fixed Tilt (Double Pitch)                        | **10**          | **11**              |
| Fixed Tilt (Conventional, ground clearance 0.5m) | **11**          | **06**              |

**As a result, the names living on the kestrel projects directory exist as the original config names, not the updated config names used in the publication nomenclature.**  
**AGAIN: the files in /projects/inspire/PySAM-MAPS/v1.2/final-backup/ use the original config names rather than the publication name**  
**The only location where files follow the updated names is after they are uploaded to the OEDI data lake (S3) with submit_s3_upload_zarrs.slurm.**  

**If you choose to regenerate files at any point using the above pipeline they will be created with the original config names then swapped to the appropriate (publication nomenclature) names only when the `scripts/submit_s3_upload_zarrs.slurm` is used**.

## inspire-agrivolt CLI
`inspire-agrivolt` defines a cli interface run the PySAM wrapper over SAM configs and postprocess them. These utilities are wrapped in scripts which allow for batch processing for dataset creation. Most users should not need to interact with the inspire_agrivolt python source to run the dataset.

The following section describes the functions provided by the CLI at the level they are called in the scripts for context.

The main driver for ground_irradiance calculations inside the package is the **ground_irradiance** function. `inspire-agrivolt` provides a thin wrapper to this functionn with the `agrivolt_ground_irradiance` command.



`agrivolt_ground_irradiance`: runs geospatial SAM simulation on NSRDB state using PySAM, PVDeg, and Dask.

    usage: agrivolt_ground_irradiance [-h] [--confs CONFS [CONFS ...]] [--workers WORKERS] [--log_dir LOG_DIR] [--port PORT] [--local-weather LOCAL_WEATHER] [--local-meta LOCAL_META]
                                    [--gids GIDS [GIDS ...]] [--downsample DOWNSAMPLE]
                                    state target_dir conf_dir

    Run Agrivoltaics Irradiance PySAM simulation for a U.S. state

    positional arguments:
    state                 State name to run analysis on as it appears in NSRDB meta data, i.e. 'Colorado'. Only used when pulling NSRDB data
    target_dir            path to base output directory
    conf_dir              path to base configs directory

    optional arguments:
    -h, --help            show this help message and exit
    --confs CONFS [CONFS ...]
                            List of config names (default: 01 through 10), STRONGLY RECOMENDED: only run one config at a time, submit multiple sbatch jobs to run multiple configs at once
    --workers WORKERS     number of dask workers (max 64 on kestrel shared partition)
    --log_dir LOG_DIR     location of dask log files
    --port PORT           dask dashboard port
    --local-weather LOCAL_WEATHER
                            Path to local weather NetCDF file
    --local-meta LOCAL_META
                            Path to local meta CSV file
    --gids GIDS [GIDS ...]
                            List of gids to use for analysis (optional). Overrides state-based selection, disables downsampling.
    --downsample DOWNSAMPLE
                            Downsample factor, from pvdeg.utiltiies.gid_downsampling(). removes half of the points on latitude and longitude axis for each n


<!-- **beds_postprocess**: runs postprocessing to calculate planting beds irradiance.
 -->
#### SAM Config Directories Structure

conf_dir can be tricky but should look like the following from the sam outputs. The model only needs the subdirectories to contain the ``pvsamv1.json`` config file for the model to work.

    path/to/dir/SAM

    ├── 01
    │   ├── 01_cashloan.json
    │   ├── 01_grid.json
    │   ├── 01.json
    │   ├── 01_pvsamv1.json
    │   ├── 01_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 02
    │   ├── 02_cashloan.json
    │   ├── 02_grid.json
    │   ├── 02.json
    │   ├── 02_pvsamv1.json
    │   ├── 02_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 03
    │   ├── 03_cashloan.json
    │   ├── 03_grid.json
    │   ├── 03.json
    │   ├── 03_pvsamv1.json
    │   ├── 03_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 04
    │   ├── 04_cashloan.json
    │   ├── 04_grid.json
    │   ├── 04.json
    │   ├── 04_pvsamv1.json
    │   ├── 04_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 05
    │   ├── 05_cashloan.json
    │   ├── 05_grid.json
    │   ├── 05.json
    │   ├── 05_pvsamv1.json
    │   ├── 05_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 06
    │   ├── 06_cashloan.json
    │   ├── 06_grid.json
    │   ├── 06.json
    │   ├── 06_pvsamv1.json
    │   ├── 06_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 07
    │   ├── 07_cashloan.json
    │   ├── 07_grid.json
    │   ├── 07.json
    │   ├── 07_pvsamv1.json
    │   ├── 07_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 08
    │   ├── 08_cashloan.json
    │   ├── 08_grid.json
    │   ├── 08.json
    │   ├── 08_pvsamv1.json
    │   ├── 08_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 09
    │   ├── 09_cashloan.json
    │   ├── 09_grid.json
    │   ├── 09.json
    │   ├── 09_pvsamv1.json
    │   ├── 09_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 10
    │   ├── 10_cashloan.json
    │   ├── 10_grid.json
    │   ├── 10.json
    │   ├── 10_pvsamv1.json
    │   ├── 10_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    └── SAM_10setups.sam


#### Example kestrel run

**There are many scripts in the scripts/ folder which can dispatch many slurm jobs at the same time for various state and configuration combinations.**

``$ agrivolt_ground_irradiance Colorado ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/Full-Outputs/Colorado/ ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM --confs 01 --port 22118``

    - calculates ground irradiance for full resolution colorado NSRDB TMY dataset
    - output_dir: ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/Full-Outputs/Colorado/
    - conf_dir: ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM 
    - confs: only run conf 01
    - port: expose dask dashboard on port 22118

<!-- 
Kestrel jobs may run out of time, If this is the case, use ``completeness.ipynb`` to determine which gids are missing then echo the gids files back into an agrivolt irradiance job. Information on how to do this coming soon...  

This is being done in `completeness.py` currently. The function `inspire_agrivolt.check_completeness` examines all possible gids in a state then checks the directory to see if they are present. If there are missing gids they will be outputted by the function, then we can use this list of gids to run the script again to produce the missing outputs. 
 -->
<!-- ./submit_fill.sh <statename> <config> "$(cat <gidsfile with space-seperated gids, no newlines>)"
 -->
<!-- After running fill-in calculations we need to combine them with the original model outputs before continuing with postprocessing. Use ``merge-fill.ipynb`` to do this. If your fills are not complete they will have to be re-run at some point. ``merge-fill.ipynb`` will allow you to silently continue with missing data.

At this point the data can be visually inspected to check for missing locations. Repeat the steps in the previous 3 paragraphs to fill in the data, if it is still missing.

Finally, we will run the beds postprocessing using ``postprocessing.ipynb``. This will create a new zarr output for each input zarr. Depending on how you want the
final format of your data, they may have to be merged or sharded later. -->