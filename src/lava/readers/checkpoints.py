"""Durable, immutable question checkpoints with strict compatibility validation."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from lava.readers.prompts import PROMPT_VERSION
from lava.readers.records import verify_benchmark_record
from lava.readers.schemas import BenchmarkRecord, OracleExample, ResolvedModel, SageMakerJobPlan


def checkpoint_contract(
    *,
    model: ResolvedModel,
    git_sha: str,
    manifest_sha: str,
    protocol_lock_id: str,
    experiment_id: str,
    limit: int,
) -> dict[str, Any]:
    """Bind reusable work to the exact code, data, model, and decoding configuration."""
    if model.generation.do_sample:
        raise ValueError("Question checkpoint reuse requires deterministic decoding")
    return {
        "schema_version": 1,
        "model": model.model_dump(mode="json"),
        "git_commit_sha": git_sha,
        "asset_manifest_sha256": manifest_sha,
        "protocol_lock_id": protocol_lock_id,
        "experiment_id": experiment_id,
        "prompt_version": PROMPT_VERSION,
        "question_count": limit,
    }


def _encode(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class QuestionCheckpoint:
    """A completed scored answer and its unmodified generation."""

    record: BenchmarkRecord
    raw_response: str


class CheckpointStore:
    """Read or create private S3 objects; never overwrite a completed answer."""

    def __init__(self, s3: Any, *, bucket: str, prefix: str, contract: dict[str, Any]):
        self.s3 = s3
        self.bucket = bucket
        self.root_prefix = prefix.rstrip("/")
        self.prefix = self.root_prefix + "/checkpoints/questions/"
        self.parent_key = self.root_prefix + "/checkpoints/parent.json"
        self.contract = contract

    def _key(self, question_id: str) -> str:
        return self.prefix + hashlib.sha256(question_id.encode()).hexdigest() + ".json"

    def _read(self, key: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as body:
            payload = body.read()
        if hashlib.sha256(payload).hexdigest() != response.get("Metadata", {}).get("sha256"):
            raise RuntimeError("Question checkpoint checksum mismatch")
        return bytes(payload)

    def save(self, checkpoint: QuestionCheckpoint) -> None:
        """Atomically persist record and raw text together, acknowledging only durable work."""
        payload = _encode(
            {
                "contract": self.contract,
                "record": checkpoint.record.model_dump(mode="json"),
                "raw_response": checkpoint.raw_response,
            }
        )
        self._put(self._key(checkpoint.record.question_id), payload)

    def link_parent(self, prefix: str) -> None:
        """Persist ancestry before copying, so repeated interruptions retain all prior work."""
        if prefix.rstrip("/") == self.root_prefix:
            raise ValueError("Checkpoint parent cannot be the current attempt")
        self._put(
            self.parent_key,
            _encode({"contract": self.contract, "parent_prefix": prefix.rstrip("/")}),
        )

    def _put(self, key: str, payload: bytes) -> None:
        digest = hashlib.sha256(payload)
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                Metadata={"sha256": digest.hexdigest()},
                ChecksumSHA256=base64.b64encode(digest.digest()).decode(),
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise
            if self._read(key) != payload:
                raise RuntimeError(
                    "Refusing to overwrite a different question checkpoint"
                ) from error

    def load(self, examples: tuple[OracleExample, ...]) -> dict[str, QuestionCheckpoint]:
        """Validate all retained work before any model is constructed or inference resumes."""
        return self._load(examples, seen=set())

    def _load(
        self, examples: tuple[OracleExample, ...], *, seen: set[str]
    ) -> dict[str, QuestionCheckpoint]:
        if self.root_prefix in seen or len(seen) >= 64:
            raise RuntimeError("Checkpoint ancestry is cyclic or exceeds 64 attempts")
        seen.add(self.root_prefix)
        expected = {example.question_id: example for example in examples}
        result: dict[str, QuestionCheckpoint] = {}
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = [
            item["Key"]
            for page in paginator.paginate(
                Bucket=self.bucket, Prefix=self.root_prefix + "/checkpoints/"
            )
            for item in page.get("Contents", [])
        ]
        if self.parent_key in keys:
            parent = json.loads(self._read(self.parent_key))
            if set(parent) != {"contract", "parent_prefix"} or parent["contract"] != self.contract:
                raise RuntimeError("Incompatible checkpoint ancestry")
            prefix = parent["parent_prefix"]
            if not isinstance(prefix, str) or not prefix or prefix.startswith("/"):
                raise ValueError("Invalid checkpoint parent prefix")
            result = CheckpointStore(
                self.s3, bucket=self.bucket, prefix=prefix, contract=self.contract
            )._load(examples, seen=seen)
        for key in keys:
            if key != self.parent_key:
                data = json.loads(self._read(key))
                if set(data) != {"contract", "record", "raw_response"}:
                    raise RuntimeError("Malformed question checkpoint")
                if data["contract"] != self.contract:
                    raise RuntimeError("Incompatible question checkpoint contract; reuse refused")
                record = BenchmarkRecord.model_validate(data["record"])
                question_id = record.question_id
                if question_id not in expected or key != self._key(question_id):
                    raise RuntimeError("Question checkpoint identity mismatch")
                for field in (
                    "git_commit_sha",
                    "asset_manifest_sha256",
                    "protocol_lock_id",
                    "experiment_id",
                    "prompt_version",
                ):
                    if getattr(record, field) != self.contract[field]:
                        raise RuntimeError(f"Question checkpoint lineage mismatch: {field}")
                model = self.contract["model"]
                for field, value in {
                    "model_key": model["model_key"],
                    "model_id": model["model_id"],
                    "model_revision": model["revision"],
                }.items():
                    if getattr(record, field) != value:
                        raise RuntimeError(f"Question checkpoint model mismatch: {field}")
                raw = data["raw_response"]
                if not isinstance(raw, str):
                    raise TypeError("Question checkpoint raw response must be text")
                verify_benchmark_record(record, expected[question_id], raw)
                checkpoint = QuestionCheckpoint(record, raw)
                if question_id in result and result[question_id] != checkpoint:
                    raise RuntimeError("Conflicting answers across checkpoint attempts")
                result[question_id] = checkpoint
        return result


def prepare_resume_plan(
    *,
    plan: SageMakerJobPlan,
    description: dict[str, Any],
    s3: Any,
    repo_root: Path,
    attempt_id: str,
) -> tuple[SageMakerJobPlan, int]:
    """Validate terminal source lineage and reusable bytes before authorizing compute."""
    from lava.readers.evaluation_contract import (
        load_evaluation_contract,
        validate_evaluation_manifest,
    )
    from lava.readers.model_registry import load_resolved_model
    from lava.readers.sagemaker_artifacts import canonical_output_s3_prefix, split_s3_uri

    if plan.mode != "benchmark":
        raise ValueError("Resume is supported only for the complete benchmark")
    if description.get("TrainingJobStatus") not in {"Failed", "Stopped"}:
        raise ValueError(
            "Resume requires a Failed or Stopped job; monitor active jobs or sync completed jobs"
        )
    params = description.get("HyperParameters", {})
    if params.get("checkpoint_schema_version") != "1":
        raise ValueError("This job predates durable question checkpoints and cannot be resumed")
    if description.get("Environment", {}).get("LAVA_GIT_COMMIT_SHA") != plan.git_commit_sha:
        raise ValueError("Resume requires the same Git commit; use the source job's exact checkout")
    if description.get("ResourceConfig", {}).get("InstanceType") != plan.instance_type:
        raise ValueError("Resume hardware mismatch")
    experiment_id = f"benchmark-{plan.model_key}-{plan.git_commit_sha[:8]}"
    for field, expected in {
        "mode": "benchmark",
        "limit": str(plan.limit),
        "model_key": plan.model_key,
        "protocol_lock_id": plan.protocol_lock_id,
        "manifest_s3_uri": plan.manifest_s3_uri,
        "experiment_id": experiment_id,
    }.items():
        if params.get(field) != expected:
            raise ValueError(f"Resume job contract mismatch: {field}")
    source_uri = canonical_output_s3_prefix(description, job_name=description["TrainingJobName"])
    base = plan.output_s3_prefix.rstrip("/")
    if not source_uri or (source_uri != base and not source_uri.startswith(base + "/attempts/")):
        raise ValueError("Resume source is outside the compatible benchmark prefix")
    if not attempt_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in attempt_id
    ):
        raise ValueError("Invalid resume attempt identifier")
    contract = load_evaluation_contract(repo_root)
    bucket, manifest_key = split_s3_uri(plan.manifest_s3_uri)
    with s3.get_object(
        Bucket=bucket, Key=manifest_key, VersionId=contract["private_manifest_version_id"]
    )["Body"] as body:
        examples = validate_evaluation_manifest(body.read(), contract)
    model = load_resolved_model(
        repo_root / "configs/oracle_reader_models.lock.json", plan.model_key
    )
    spec = checkpoint_contract(
        model=model,
        git_sha=plan.git_commit_sha,
        manifest_sha=contract["private_manifest_sha256"],
        protocol_lock_id=plan.protocol_lock_id,
        experiment_id=experiment_id,
        limit=plan.limit,
    )
    _, source_key = split_s3_uri(source_uri)
    saved = CheckpointStore(s3, bucket=bucket, prefix=source_key, contract=spec).load(examples)
    updated = SageMakerJobPlan.model_validate(
        {
            **plan.model_dump(),
            "output_s3_prefix": base + "/attempts/" + attempt_id,
            "resume_s3_prefix": source_uri,
        }
    )
    return updated, len(saved)
