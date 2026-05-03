import argparse
import boto3

def delete_prefix(bucket: str, prefix: str, *, dry_run: bool = False) -> int:
    """
    Deletes all objects whose keys start with `prefix`.
    Returns number of objects targeted for deletion.

    Note: If bucket versioning is enabled, this deletes the *current* versions
    by placing delete markers; old versions remain unless you delete versions too.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    total = 0
    batch = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            batch.append({"Key": obj["Key"]})
            total += 1

            # DeleteObjects max = 1000 keys per request
            if len(batch) == 1000:
                if not dry_run:
                    s3.delete_objects(
                        Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
                    )
                batch.clear()

    if batch:
        if not dry_run:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
        batch.clear()

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="s3_delete",
        description="Delete all objects under an S3 prefix (recursive).",
    )

    parser.add_argument(
        "prefix",
        type=str,
        help="S3 key prefix to remove (e.g. 'inspire/my-zarr/'). Do NOT include s3://",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="oedi-data-drop",
        help="S3 bucket name (e.g. 'oedi-data-drop'). Do NOT include s3://",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List/count what would be deleted, but do not delete.",
    )

    args = parser.parse_args()

    n = delete_prefix(args.bucket, args.prefix, dry_run=args.dry_run)
    if args.dry_run:
        print("Would delete:", n)
    else:
        print("Deleted:", n)


if __name__ == "__main__":
    main()

