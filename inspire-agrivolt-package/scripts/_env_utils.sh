#!/usr/bin/env bash

# If this file is executed directly, $0 == ${BASH_SOURCE[0]}.
# If it is sourced, they are different.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cat <<EOF >&2
[ERROR] This is a helper library and is not meant to be executed directly.
       Source it from another script, for example:

       source scripts/lib/env_utils.sh
EOF
    exit 1
fi

get_inspire_agrivolt_conda_env() {
    if [[ -z "${INSPIRE_AGRIVOLT_CONDA_ENV:-}" ]]; then
        cat <<EOF >&2
[ERROR] INSPIRE_AGRIVOLT_CONDA_ENV is not set.

Please set it to the name or path of the conda/mamba environment, e.g.:

  export INSPIRE_AGRIVOLT_CONDA_ENV=/projects/inspire/envs/inspire-agrivolt-geospatial
  # or
  export INSPIRE_AGRIVOLT_CONDA_ENV=geospatial

Then re-run your sbatch command.
EOF
        exit 1
    fi

    printf '%s\n' "$INSPIRE_AGRIVOLT_CONDA_ENV"
}

get_inspire_agrivolt_SAM_conf_dir() {
    if [[ -z "${INSPIRE_AGRIVOLT_SAM_CONFIG_DIR:-}" ]]; then
        cat <<EOF >&2
[ERROR] INSPIRE_AGRIVOLT_SAM_CONFIG_DIR is not set.

Please set it to the path of the SAM configuration directiory, e.g.:

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

Then re-run your sbatch command.
EOF
        exit 1
    fi

    printf '%s\n' "$INSPIRE_AGRIVOLT_SAM_CONFIG_DIR"
}

get_inspire_agrivolt_model_outs_dir() {
    if [[ -z "${INSPIRE_AGRIVOLT_MODEL_OUTS_DIR:-}" ]]; then
        cat <<EOF >&2
[ERROR] INSPIRE_AGRIVOLT_MODEL_OUTS_DIR is not set.

Set to the path of the desired agrivoltaic irradiance outputs directory, e.g:

    export INSPIRE_AGRIVOLT_MODEL_OUTS_DIR=/projects/inspire/PySAM-MAPS/v1.2/model-outs/

    This is where the outputs of the irradiance model go. 
    Produced by the following scripts in `scripts/`:
    - `submit_state_conf.sh`
    - `run_all_configs_state.sh`
    - `run_all_states_config.sh`


Then re-run your sbatch command.
EOF
        exit 1
    fi

    printf '%s\n' "$INSPIRE_AGRIVOLT_MODEL_OUTS_DIR"
}


get_inspire_agrivolt_postprocess_dir() {
    if [[ -z "${INSPIRE_AGRIVOLT_POSTPROCESS_DIR:-}" ]]; then
        cat <<EOF >&2
[ERROR] INSPIRE_AGRIVOLT_POSTPROCESS_DIR is not set.

Please set to the path of the desired postprocessing outputs directory, e.g:

    export INSPIRE_AGRIVOLT_POSTPROCESS_DIR=/projects/inspire/PySAM-MAPS/v1.2/postprocess/

    This is where the outputs of the postprocessing step go. Produced by the following script in `scripts/`:
    - `submit_postprocess.slurm`

Then re-run your sbatch command.
EOF
        exit 1
    fi

    printf '%s\n' "$INSPIRE_AGRIVOLT_POSTPROCESS_DIR"
}


get_inspire_agrivolt_final_dir() {
    if [[ -z "${INSPIRE_AGRIVOLT_POSTPROCESS_DIR:-}" ]]; then
        cat <<EOF >&2
[ERROR] INSPIRE_AGRIVOLT_FINAL_DIR is not set.


Please set to the path of the desired final combined outputs directory, e.g:

    export INSPIRE_AGRIVOLT_FINAL_DIR=/projects/inspire/PySAM-MAPS/v1.2/final/

This is where the final outputs of the combine step are stored. This step takes results from `INSPIRE_AGRIVOLT_MODEL_OUTS_DIR` and `INSPIRE_AGRIVOLT_POSTPROCESSING_DIR` and combined them into their final state. No more processing is done on the combined files in `INSPIRE_AGRIVOLT_FINAL_DIR`. They are produced by the following script in `scripts/`:
- `submit_combine.slurm`.

Then re-run your sbatch command.
EOF
        exit 1
    fi

    printf '%s\n' "$INSPIRE_AGRIVOLT_FINAL_DIR"
}
