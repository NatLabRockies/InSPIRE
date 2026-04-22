"""
Examine the inspire-agrivolt-package README.md for this table and more details in section #4 - Deployment
All code in every step of the pipeline on Kestrel (before Deployment/Upload to S3) references the original config names. 

As a result, the names living on the kestrel projects directory exist as the original config names, not the updated config names used in the publication nomenclature.
AGAIN: the files in /projects/inspire/PySAM-MAPS/v1.2/final-backup/ use the original config names rather than the publication names.
The only location where files follow the updated names is after they are uploaded to the OEDI data lake (S3) with submit_s3_upload_zarrs.slurm.

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
"""
 
# kestrel name : published name
KESTREL_NAME_TO_PUBLISHED_NAME_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 6,
}

PUBLISHED_NAME_TO_KESTREL_NAME_MAP = {value: key for key, value in KESTREL_NAME_TO_PUBLISHED_NAME_MAP.items()}

def convert_published_name_to_kestrel_name(published_name: int) -> int:
    """Take published (new) config name/number and convert it to the old config name/number"""
    return PUBLISHED_NAME_TO_KESTREL_NAME_MAP[published_name]

def convert_kestrel_name_to_published_name(kestrel_name: int) -> int:
    """Take old config name/number and convert it to the published (new) config name/number """
    return KESTREL_NAME_TO_PUBLISHED_NAME_MAP[kestrel_name]
