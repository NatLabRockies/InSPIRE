"""
Script for parallelizing uploads from Kestrel to S3.
"""

import os
import sys
import logging
import argparse
from collections.abc import Iterable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

OEDI_STAGING_BUCKET = "oedi-data-drop"
HTTP_MAX_POOL_CONNECTIONS = 64
MULTIPROCESSING_MAX_WORKERS = 32

MULTIPART_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MULTIPART_THRESHOLD = 32 * 1024 * 1024
MULTIPART_CHUNK = 64 * 1024 * 1024

logger = logging.getLogger("uploader")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def iter_local_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def build_key(root: Path, file: Path, inspire_prefix: str) -> str:
    rel = file.relative_to(root).as_posix()
    return f"{inspire_prefix}{rel}"


def upload_one(s3_client, file: Path, bucket: str, key: str, transfer_cfg: TransferConfig) -> tuple[str, int]:
    s3_client.upload_file(
        Filename=str(file),
        Bucket=bucket,
        Key=key,
        Config=transfer_cfg,
        ExtraArgs={},
    )
    return key, file.stat().st_size


def rotated_s3_name(zarr_name: str) -> str:
    """
    Rotate only names 06..11:
      06->07, 07->08, 08->09, 09->10, 10->11, 11->06
    Otherwise unchanged.
    """
    mapping = {
        "06.zarr": "07.zarr",
        "07.zarr": "08.zarr",
        "08.zarr": "09.zarr",
        "09.zarr": "10.zarr",
        "10.zarr": "11.zarr",
        "11.zarr": "06.zarr",
    }
    return mapping.get(zarr_name, zarr_name)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("local_zarr_name", type=str, help="local zarr dir name, e.g. 06.zarr")
    parser.add_argument("final_dir", type=str, help="directory containing final merged zarrs")

    parser.add_argument(
        "--rotate-prefix",
        action="store_true",
        help="Do NOT rename locally; only rotate S3 prefix names for 06..11 (11 uploads under 06, 06 under 07, ...).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan but do not upload.")

    args = parser.parse_args()

    local_zarr = args.local_zarr_name
    final_dir = Path(args.final_dir)

    local_root = final_dir / local_zarr
    if not local_root.is_dir():
        raise NotADirectoryError(f"Local zarr dir not found: {local_root}")

    # Decide what the zarr should be CALLED on S3
    s3_zarr = rotated_s3_name(local_zarr) if args.rotate_prefix else local_zarr

    inspire_prefix = f"inspire/{s3_zarr}/"
    logger.info("Local: %s  ->  S3 prefix: %s", local_zarr, inspire_prefix)

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY env vars")

    s3_client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=HTTP_MAX_POOL_CONNECTIONS,
            retries={"mode": "adaptive", "max_attempts": 10},
        ),
        region_name="us-west-2",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    transfer_cfg = TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD,
        multipart_chunksize=MULTIPART_CHUNK,
        max_concurrency=MULTIPART_MAX_WORKERS,
        use_threads=True,
    )

    to_upload: list[tuple[Path, str, int]] = []
    total_bytes = 0
    for f in iter_local_files(local_root):
        key = build_key(local_root, f, inspire_prefix=inspire_prefix)
        size = f.stat().st_size
        to_upload.append((f, key, size))
        total_bytes += size

    logger.info(
        "Planned %d uploads (%.2f GiB). Starting with up to %d workers...",
        len(to_upload),
        total_bytes / 1024 / 1024 / 1024,
        MULTIPROCESSING_MAX_WORKERS,
    )

    if args.dry_run:
        logger.info("DRY RUN: skipping uploads.")
        return

    errors = 0
    futures = []
    with ThreadPoolExecutor(max_workers=MULTIPROCESSING_MAX_WORKERS) as pool:
        for f, key, _ in to_upload:
            futures.append(pool.submit(upload_one, s3_client, f, OEDI_STAGING_BUCKET, key, transfer_cfg))

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                key, sz = fut.result()
                if i % 100 == 0 or i == len(futures):
                    logger.info("Progress: %d/%d (last: %s, %d bytes)", i, len(futures), key, sz)
            except ClientError as e:
                errors += 1
                logger.error("Upload failed: %s", e, exc_info=False)

    logger.info("Done. Successful: %d  Failed: %d", len(futures) - errors, errors)


if __name__ == "__main__":
    main()

