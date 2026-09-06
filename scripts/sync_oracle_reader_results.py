"""Verify and persist a completed oracle-reader public summary into the repository."""

from __future__ import annotations

import argparse
import hashlib
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
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.sagemaker_artifacts import (
    canonical_output_s3_prefix,
    split_s3_uri,
)


def _log(event: str, started: float, **fields: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True))


def _latest_completed_oracle_job(client: Any) -> str:
    response = client.list_training_jobs(
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=50,
    )

    for item in response.get("TrainingJobSummaries", []):
        name = item.get("TrainingJobName", "")
        if isinstance(name, str) and "lava-oracle" in name:
            return name

    raise RuntimeError("No completed LAVA oracle-reader SageMaker job was found")


def _read_body(response: Mapping[str, object], *, label: str) -> bytes:
    body = response.get("Body")

    if body is None or not hasattr(body, "read"):
        raise RuntimeError(f"{label} returned no readable body")

    payload = body.read()

    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError(f"{label} is empty or malformed")

    return payload


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sync() -> int:
    """Sync one verified public summary without shell-exported bucket state."""
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

    job_name = args.job_name or _latest_completed_oracle_job(sagemaker)
    _log("results.sync.started", started, job_name=job_name, region=region)

    description = sagemaker.describe_training_job(TrainingJobName=job_name)
    description_mapping = cast(Mapping[str, object], description)

    if description_mapping.get("TrainingJobStatus") != "Completed":
        raise RuntimeError("Result sync requires a Completed SageMaker training job")

    output_prefix = canonical_output_s3_prefix(
        description_mapping,
        job_name=job_name,
    )

    if output_prefix is None:
        raise RuntimeError(
            "Result sync requires a canonical output prefix; legacy archive-only "
            "runs are not eligible for public-summary synchronization."
        )

    gate = verify_training_model_artifact(
        sagemaker_client=sagemaker,
        s3_client=s3,
        job_name=job_name,
        expected_output_s3_prefix=output_prefix,
    )
    _log(
        "results.sync.artifact_verified",
        started,
        job_name=job_name,
        artifact_gate=gate.as_dict(),
    )

    bucket, prefix = split_s3_uri(output_prefix)
    key = f"{prefix}/public_summary.json"
    response = cast(
        Mapping[str, object],
        s3.get_object(Bucket=bucket, Key=key),
    )
    payload = _read_body(
        response,
        label="canonical public_summary.json",
    )

    metadata = response.get("Metadata", {})
    expected_sha256: str | None = None

    if isinstance(metadata, Mapping):
        candidate = metadata.get("sha256")
        if isinstance(candidate, str) and candidate:
            expected_sha256 = candidate

    observed_sha256 = hashlib.sha256(payload).hexdigest()

    if expected_sha256 is not None and expected_sha256 != observed_sha256:
        raise RuntimeError(
            "Public summary checksum did not match S3 metadata: "
            f"expected={expected_sha256!r}, observed={observed_sha256!r}"
        )

    try:
        summary = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("public_summary.json is not valid UTF-8 JSON") from exc

    if not isinstance(summary, dict):
        raise TypeError("public_summary.json must contain one JSON object")

    run_dir = root / "reports" / "oracle_reader" / "runs" / job_name
    run_summary = run_dir / "public_summary.json"
    latest_summary = root / "reports" / "oracle_reader" / "latest_public_summary.json"

    _write_atomic(run_summary, payload)
    _write_atomic(latest_summary, payload)

    resource_config = description_mapping.get("ResourceConfig", {})
    instance_type: str | None = None

    if isinstance(resource_config, Mapping):
        candidate_instance = resource_config.get("InstanceType")
        if isinstance(candidate_instance, str):
            instance_type = candidate_instance

    manifest = {
        "schema_version": 1,
        "synced_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "job_name": job_name,
        "model_key": summary.get("model_key"),
        "model_id": summary.get("model_id"),
        "model_revision": summary.get("model_revision"),
        "git_commit_sha": summary.get("git_commit_sha"),
        "protocol_lock_id": summary.get("protocol_lock_id"),
        "instance_type": instance_type,
        "training_time_seconds": description_mapping.get("TrainingTimeInSeconds"),
        "billable_time_seconds": description_mapping.get("BillableTimeInSeconds"),
        "public_summary_sha256": observed_sha256,
        "artifact_gate": gate.as_dict(),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_atomic(run_dir / "sync_manifest.json", manifest_bytes)

    _log(
        "results.sync.completed",
        started,
        job_name=job_name,
        run_summary=str(run_summary.relative_to(root)),
        latest_summary=str(latest_summary.relative_to(root)),
        public_summary_sha256=observed_sha256,
    )
    print(f"PUBLIC_SUMMARY_SHA256={observed_sha256}")
    print(run_summary)
    print(latest_summary)
    print("ORACLE_READER_RESULTS_SYNCED")
    return 0


def main() -> int:
    """Keep artifact verification and S3 reads observable throughout synchronization."""
    logger = RuntimeEventLogger("oracle_reader.sync")
    with logger.stage("sync", heartbeat_seconds=15.0):
        return _sync()


if __name__ == "__main__":
    raise SystemExit(main())
