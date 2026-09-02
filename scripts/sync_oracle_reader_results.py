"""Download and validate the latest public oracle-reader summary from private S3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import boto3


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / "artifacts/oracle_reader/latest_job.json"
    if not state_path.exists():
        raise SystemExit("No submitted job state exists yet")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = state["plan"]
    bucket = os.environ.get("S3_BUCKET")
    if not bucket or bucket != plan["bucket"]:
        raise SystemExit("S3_BUCKET does not match the submitted job plan")
    prefix = plan["output_s3_prefix"].removeprefix(f"s3://{bucket}/").rstrip("/")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    response = s3.get_object(Bucket=bucket, Key=f"{prefix}/public_summary.json")
    payload = response["Body"].read()
    expected = response.get("Metadata", {}).get("sha256")
    observed = hashlib.sha256(payload).hexdigest()
    if expected and expected != observed:
        raise SystemExit("Public summary checksum did not match S3 metadata")
    output = repo_root / "reports/oracle_reader/latest_public_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"PUBLIC_SUMMARY_SHA256={observed}")
    print(output)


if __name__ == "__main__":
    main()
