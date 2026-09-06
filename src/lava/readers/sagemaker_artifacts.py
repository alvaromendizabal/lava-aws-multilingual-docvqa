"""Shared SageMaker artifact-location helpers for oracle-reader tooling."""

from __future__ import annotations

from collections.abc import Mapping


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split a non-empty S3 URI into bucket and key."""
    if not uri.startswith("s3://"):
        raise RuntimeError(f"Expected S3 URI, observed {uri!r}")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")

    if not separator or not bucket or not key:
        raise RuntimeError(f"Malformed S3 URI: {uri!r}")

    return bucket, key.rstrip("/")


def model_artifact_uri(description: Mapping[str, object]) -> str:
    """Return the SageMaker model artifact URI from a training-job description."""
    artifacts = description.get("ModelArtifacts", {})

    if not isinstance(artifacts, Mapping):
        raise TypeError("Training job returned malformed ModelArtifacts")

    uri = artifacts.get("S3ModelArtifacts")

    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise RuntimeError("Training job has no valid S3 model artifact URI")

    return uri.rstrip("/")


def canonical_output_s3_prefix(
    description: Mapping[str, object],
    *,
    job_name: str,
) -> str | None:
    """Derive the canonical benchmark output prefix from SageMaker metadata."""
    output_config = description.get("OutputDataConfig", {})

    if not isinstance(output_config, Mapping):
        raise TypeError("Training job returned malformed OutputDataConfig")

    compression_type = output_config.get("CompressionType")
    raw_output_path = output_config.get("S3OutputPath")

    if isinstance(raw_output_path, str) and raw_output_path.startswith("s3://"):
        output_path = raw_output_path.rstrip("/")
        suffix = "/sagemaker-output"

        if output_path.endswith(suffix):
            return output_path[: -len(suffix)]

    artifact_uri = model_artifact_uri(description)
    artifact_suffix = f"/sagemaker-output/{job_name}/output/model"

    if artifact_uri.endswith(artifact_suffix):
        return artifact_uri[: -len(artifact_suffix)]

    if compression_type == "NONE":
        raise RuntimeError(
            "Unable to derive the canonical output prefix for an uncompressed "
            "SageMaker artifact. Refusing partial artifact verification."
        )

    return None
