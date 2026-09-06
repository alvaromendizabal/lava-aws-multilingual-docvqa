"""Verified source caching and immutable submission bundles in S3."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from lava.evaluation.semantic import ImmutableS3Objects, encode
from lava.evaluation.submission import sha256
from lava.readers.runtime_logging import RuntimeEventLogger


def atomic_write(path: Path, payload: bytes) -> None:
    """Install a complete file after flushing its content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_source(
    s3: Any, bucket: str, path: Path, source: dict[str, str], logger: RuntimeEventLogger
) -> bytes:
    """Reuse only checksum-valid bytes; restore a missing or corrupt local cache from S3."""
    if path.exists():
        payload = path.read_bytes()
        if sha256(payload) == source["sha256"]:
            logger.emit("submission.input.reused", filename=path.name, byte_count=len(payload))
            return payload
        logger.emit("submission.input.cache_rejected", filename=path.name)
    response = s3.get_object(Bucket=bucket, Key=source["key"], VersionId=source["version_id"])
    with response["Body"] as stream:
        payload = stream.read()
    if response.get("VersionId") != source["version_id"] or sha256(payload) != source["sha256"]:
        raise ValueError("S3 source version or checksum differs from the submission contract")
    atomic_write(path, payload)
    logger.emit("submission.input.persisted", filename=path.name, byte_count=len(payload))
    return payload


def persist_submission(
    s3: Any,
    bucket: str,
    root: Path,
    payload: bytes,
    manifest: dict[str, Any],
    logger: RuntimeEventLogger,
) -> Path:
    """Persist CSV first and the commit manifest last; identical retries safely resume."""
    if manifest.get("submission_sha256") != sha256(payload):
        raise ValueError("Submission manifest does not identify the supplied CSV")
    identity = sha256(encode(manifest))
    prefix = f"experiments/submissions/lava-challenge-2026/{identity}"
    key = prefix + "/submission.csv"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="text/csv; charset=utf-8",
            IfNoneMatch="*",
            Metadata={"sha256": sha256(payload)},
            ChecksumSHA256=base64.b64encode(hashlib.sha256(payload).digest()).decode(),
        )
        logger.emit("submission.csv.persisted", bundle_id=identity, byte_count=len(payload))
    except ClientError as error:
        # S3 has no modeled exception class for conditional-write precondition failure.
        if error.response.get("Error", {}).get("Code") not in {"412", "PreconditionFailed"}:
            raise
        response = s3.get_object(Bucket=bucket, Key=key)
        with response["Body"] as stream:
            existing = stream.read()
        if existing != payload or response.get("Metadata", {}).get("sha256") != sha256(payload):
            raise ValueError("Conflicting or corrupted immutable submission CSV") from None
        logger.emit("submission.csv.reused", bundle_id=identity)
    ImmutableS3Objects(s3, bucket, prefix).write("manifest.json", manifest)
    target = root / "artifacts/submission" / identity / "submission.csv"
    atomic_write(target, payload)
    atomic_write(target.parent / "manifest.json", encode(manifest))
    logger.emit("submission.bundle.completed", bundle_id=identity, uploaded_to_kaggle=False)
    return target
