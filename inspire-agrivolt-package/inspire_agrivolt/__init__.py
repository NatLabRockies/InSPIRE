from importlib.metadata import version
import sys
import logging

logger = logging.getLogger("inspire_agrivolt")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import main
from . import irradiance_sam
from . import beds_postprocessing

from .file_operations import (
    check_completeness,  # check for outputs at all gids in a state, dict output
    generate_missing_gids_file,
    merge_original_fill_data_to_zarr,  # merge original and fill data into a single zarr store
    check_ncs_coords_dims,  # fails if time decode fails
)
# from .verify import verify_dataset_gids

from . import verify

__version__ = version("inspire_agrivolt")
