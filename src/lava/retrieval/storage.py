"""Conditional S3 checkpoints using only GetObject and PutObject permissions."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from botocore.exceptions import ClientError

from lava.evaluation.semantic import encode


class CheckpointStore:
    """Atomically transition an empty slot to one immutable completed value.

    Creating a slot before reading avoids relying on ListBucket to distinguish
    missing keys from denied access. A crash leaves a harmless pending slot.
    Conditional ETag writes prevent competing processes overwriting completed work.
    """

    def __init__(self, client: Any, bucket: str, prefix: str):
        self.client, self.bucket, self.prefix = client, bucket, prefix.rstrip("/")

    def _put(self, key: str, payload: bytes, **conditions: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            Metadata={"sha256": hashlib.sha256(payload).hexdigest()},
            ChecksumSHA256=base64.b64encode(hashlib.sha256(payload).digest()).decode(),
            **conditions,
        )

    def _get(self, key: str) -> tuple[dict[str, Any], str]:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != response.get("Metadata", {}).get("sha256"):
            raise ValueError("Retrieval S3 checkpoint checksum mismatch")
        envelope = json.loads(payload)
        if not isinstance(envelope, dict) or envelope.get("state") not in {"pending", "complete"}:
            raise ValueError("Invalid retrieval checkpoint state")
        return envelope, response["ETag"]

    @staticmethod
    def _precondition(error: ClientError) -> bool:
        # S3 has no modeled exception type for conditional-write conflicts.
        return error.response.get("Error", {}).get("Code") in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
            "412",
            "409",
        }

    def read(self, suffix: str) -> dict[str, Any] | None:
        """Ensure a slot exists, then read verified completion or a pending state."""
        key = f"{self.prefix}/{suffix}"
        try:
            self._put(key, encode({"state": "pending"}), IfNoneMatch="*")
        except ClientError as error:
            if not self._precondition(error):
                raise
        envelope, _ = self._get(key)
        if envelope["state"] == "pending":
            return None
        value = envelope.get("value")
        if not isinstance(value, dict):
            raise TypeError("Completed retrieval checkpoint needs a JSON object")
        return value

    def write(self, suffix: str, value: dict[str, Any]) -> None:
        """Commit a pending slot with compare-and-swap; reject conflicting completions."""
        # Callers normally read first; allow standalone publication too.
        existing = self.read(suffix)
        if existing is not None:
            if existing != value:
                raise ValueError("Conflicting completed retrieval checkpoint")
            return
        key = f"{self.prefix}/{suffix}"
        current, etag = self._get(key)
        if current["state"] == "complete":
            if current.get("value") != value:
                raise ValueError("Conflicting completed retrieval checkpoint")
            return
        try:
            self._put(key, encode({"state": "complete", "value": value}), IfMatch=etag)
        except ClientError as error:
            if not self._precondition(error):
                raise
            if self.read(suffix) != value:
                raise ValueError("Concurrent retrieval checkpoint conflict") from error
