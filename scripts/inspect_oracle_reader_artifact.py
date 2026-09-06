"""Inspect a completed oracle-reader SageMaker artifact without creating compute."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root
from lava.readers.artifact_gate import verify_training_model_artifact
from lava.readers.sagemaker_artifacts import (
    canonical_output_s3_prefix,
    model_artifact_uri,
)


def _log(event: str, started: float, **fields: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True))


def _latest_oracle_job(client: Any) -> str:
    response = client.list_training_jobs(
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
    """Verify a completed job using its actual compressed or uncompressed layout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name")
    args = parser.parse_args()

    started = time.monotonic()
    root = find_repo_root(Path(__file__).resolve())
    load_dotenv(root / ".env", override=False)

    region = os.environ.get("AWS_REGION", "us-west-2")
    session = boto3.session.Session(region_name=region)
    sagemaker = session.client("sagemaker")
    s3 = session.client("s3")

    job_name = args.job_name or _latest_oracle_job(sagemaker)
    _log("artifact.inspect.started", started, job_name=job_name, region=region)

    description = sagemaker.describe_training_job(TrainingJobName=job_name)
    description_mapping = cast(Mapping[str, object], description)
    status = description_mapping.get("TrainingJobStatus")

    if status != "Completed":
        raise RuntimeError(
            "Artifact inspection requires a Completed SageMaker training job; "
            f"observed status={status!r}."
        )

    artifact_uri = model_artifact_uri(description_mapping)
    output_config = description_mapping.get("OutputDataConfig", {})

    if not isinstance(output_config, Mapping):
        raise TypeError("Training job returned malformed OutputDataConfig")

    compression_type = output_config.get("CompressionType")
    expected_output_s3_prefix = canonical_output_s3_prefix(
        description_mapping,
        job_name=job_name,
    )

    _log(
        "artifact.verify.started",
        started,
        job_name=job_name,
        artifact_uri=artifact_uri,
        compression_type=compression_type,
        expected_output_s3_prefix=expected_output_s3_prefix,
    )

    report = verify_training_model_artifact(
        sagemaker_client=sagemaker,
        s3_client=s3,
        job_name=job_name,
        expected_output_s3_prefix=expected_output_s3_prefix,
    )

    payload = report.as_dict()
    _log(
        "artifact.inspect.completed",
        started,
        job_name=job_name,
        report=payload,
        verified=True,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ORACLE_READER_ARTIFACT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
