"""Fail-closed verification of private raw responses in a SageMaker model artifact."""

from __future__ import annotations

import io
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lava.readers.structured_output import parse_structured_output


@dataclass(frozen=True, slots=True)
class ArtifactGateReport:
    """Verification report produced only after an artifact passes the schema gate."""

    raw_response_count: int
    schema_valid_rate: float
    parser_error_counts: dict[str, int]
    model_artifact_uri: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""
        return asdict(self)


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in model artifact: {member.name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(
                f"Links are not allowed in model artifact verification: {member.name!r}"
            )
    return members


def _read_raw_responses_from_tar(data: bytes) -> list[str]:
    responses: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in _safe_members(archive):
            path = PurePosixPath(member.name)
            if not member.isfile():
                continue
            if "private" not in path.parts or "raw_responses" not in path.parts:
                continue
            if path.suffix != ".txt":
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Unable to read {member.name!r} from model artifact")
            responses.append(handle.read().decode("utf-8"))
    return responses


def verify_model_artifact_bytes(
    data: bytes, *, model_artifact_uri: str = "memory://artifact"
) -> ArtifactGateReport:
    """Require exactly one raw response and require it to pass the strict JSON schema."""
    if not data:
        raise RuntimeError("SageMaker model artifact is empty")
    responses = _read_raw_responses_from_tar(data)
    if len(responses) != 1:
        raise RuntimeError(f"Expected exactly one private raw response; found {len(responses)}")
    result = parse_structured_output(responses[0])
    if not result.valid:
        raise RuntimeError(f"Raw response failed structured-output gate: {result.error}")
    return ArtifactGateReport(
        raw_response_count=1,
        schema_valid_rate=1.0,
        parser_error_counts={},
        model_artifact_uri=model_artifact_uri,
    )


def verify_training_model_artifact(
    *,
    sagemaker_client: Any,
    s3_client: Any,
    job_name: str,
) -> ArtifactGateReport:
    """Download the completed training model artifact and apply the strict response gate."""
    description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    if description.get("TrainingJobStatus") != "Completed":
        raise RuntimeError("Artifact verification requires a Completed SageMaker training job")
    artifact_uri = description.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not isinstance(artifact_uri, str) or not artifact_uri.startswith("s3://"):
        raise RuntimeError("Completed training job did not expose an S3 model artifact URI")
    bucket_and_key = artifact_uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not separator or not bucket or not key:
        raise RuntimeError("Malformed SageMaker model artifact URI")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
        s3_client.download_file(bucket, key, handle.name)
        data = Path(handle.name).read_bytes()
    return verify_model_artifact_bytes(data, model_artifact_uri=artifact_uri)


def inspect_local_model_artifact(path: Path) -> dict[str, object]:
    """Inspect a local model archive for debugging without creating any cloud resource."""
    try:
        report = verify_model_artifact_bytes(path.read_bytes(), model_artifact_uri=str(path))
    except RuntimeError as exc:
        return {"verified": False, "error": str(exc), "artifact": str(path)}
    return {"verified": True, **report.as_dict()}
