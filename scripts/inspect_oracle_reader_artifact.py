"""Inspect a completed oracle-reader SageMaker model artifact without creating compute."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root
from lava.readers.artifact_gate import inspect_local_model_artifact


def _log(event: str, started: float, **fields: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True))


def _latest_oracle_job(client: object) -> str:
    response = client.list_training_jobs(  # type: ignore[attr-defined]
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    )
    for item in response.get("TrainingJobSummaries", []):
        name = item.get("TrainingJobName", "")
        if isinstance(name, str) and "lava-oracle" in name:
            return name
    raise RuntimeError("No LAVA oracle-reader SageMaker training job was found")


def main() -> int:
    """Download only the model artifact and inspect private raw-response evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name")
    args = parser.parse_args()
    started = time.monotonic()
    root = find_repo_root(Path(__file__).resolve())
    load_dotenv(root / ".env", override=False)
    region = os.environ.get("AWS_REGION", "us-west-2")
    session = boto3.session.Session(region_name=region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")
    job_name = args.job_name or _latest_oracle_job(sm)
    _log("artifact.inspect.started", started, job_name=job_name)
    description = sm.describe_training_job(TrainingJobName=job_name)
    artifact_uri = description.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not isinstance(artifact_uri, str) or not artifact_uri.startswith("s3://"):
        raise RuntimeError("Training job has no model artifact URI")
    bucket_key = artifact_uri.removeprefix("s3://")
    bucket, _, key = bucket_key.partition("/")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
        _log("artifact.download.started", started, job_name=job_name)
        s3.download_file(bucket, key, handle.name)
        report = inspect_local_model_artifact(Path(handle.name))
    _log("artifact.inspect.completed", started, job_name=job_name, report=report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("verified") and "found 0" in str(report.get("error")):
        print("RAW_RESPONSE_NOT_AVAILABLE_FOR_LEGACY_RUN")
        return 0
    if not report.get("verified"):
        print("ORACLE_READER_ARTIFACT_NOT_VERIFIED")
        return 2
    print("ORACLE_READER_ARTIFACT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
