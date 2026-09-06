"""Fail-closed verification of oracle-reader SageMaker run artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lava.readers.structured_output import parse_structured_output


@dataclass(frozen=True, slots=True)
class ArtifactGateReport:
    """Verification report produced only after run artifacts pass every gate."""

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


def _verify_one_raw_response(
    responses: list[str],
    *,
    model_artifact_uri: str,
) -> ArtifactGateReport:
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


def verify_model_artifact_bytes(
    data: bytes,
    *,
    model_artifact_uri: str = "memory://artifact",
) -> ArtifactGateReport:
    """Verify a legacy gzip model archive without creating cloud resources."""
    if not data:
        raise RuntimeError("SageMaker model artifact is empty")

    responses = _read_raw_responses_from_tar(data)

    return _verify_one_raw_response(
        responses,
        model_artifact_uri=model_artifact_uri,
    )


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise RuntimeError(f"Expected S3 URI, observed {uri!r}")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")

    if not separator or not bucket or not key:
        raise RuntimeError(f"Malformed S3 URI: {uri!r}")

    return bucket, key.rstrip("/")


def _read_s3_bytes(
    *,
    s3_client: Any,
    bucket: str,
    key: str,
    allow_empty: bool = False,
) -> bytes:
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    body = response.get("Body")

    if body is None or not hasattr(body, "read"):
        raise RuntimeError(f"S3 object s3://{bucket}/{key} returned no readable body")

    try:
        data = body.read()
    finally:
        body.close()

    if not isinstance(data, bytes) or (not data and not allow_empty):
        raise RuntimeError(f"S3 object s3://{bucket}/{key} is empty or malformed")

    return data


def _list_s3_keys(
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }

        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**kwargs)

        for item in response.get("Contents", []):
            key = item.get("Key")

            if isinstance(key, str):
                keys.append(key)

        if not response.get("IsTruncated"):
            break

        token = response.get("NextContinuationToken")

        if not isinstance(token, str) or not token:
            raise RuntimeError("S3 listing was truncated without a continuation token")

        continuation_token = token

    return sorted(keys)


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = data.decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain one JSON object")

    return payload


def _require_numeric(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
) -> float:
    value = payload.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label}.{key} must be numeric; observed {value!r}")

    return float(value)


def _verify_public_summary(
    data: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    summary = _json_object(data, label=label)

    if summary.get("record_count") != 1:
        raise RuntimeError(
            f"{label}.record_count must equal 1; observed {summary.get('record_count')!r}"
        )

    schema_valid_rate = _require_numeric(
        summary,
        "schema_valid_rate",
        label=label,
    )

    if schema_valid_rate != 1.0:
        raise RuntimeError(
            f"{label}.schema_valid_rate must equal 1.0; observed {schema_valid_rate}"
        )

    abstention_rate = _require_numeric(
        summary,
        "abstention_rate",
        label=label,
    )

    if abstention_rate != 0.0:
        raise RuntimeError(f"{label}.abstention_rate must equal 0.0; observed {abstention_rate}")

    counts = summary.get("parser_error_counts")

    if not isinstance(counts, dict):
        raise TypeError(f"{label}.parser_error_counts must be a JSON object")

    normalized_counts: dict[str, int] = {}

    for name, value in counts.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise RuntimeError(f"{label}.parser_error_counts is malformed")

        normalized_counts[name] = value

    if normalized_counts.get("none") != 1:
        raise RuntimeError(
            f"{label}.parser_error_counts must contain none=1; observed {normalized_counts!r}"
        )

    real_errors = {
        name: count for name, count in normalized_counts.items() if name != "none" and count != 0
    }

    if real_errors:
        raise RuntimeError(f"{label} reports parser errors: {real_errors}")

    return summary


def _verify_uncompressed_model_prefix(
    *,
    s3_client: Any,
    artifact_uri: str,
) -> tuple[
    ArtifactGateReport,
    str,
    str,
    dict[str, Any],
]:
    """Verify an uncompressed SageMaker model-output prefix and raw-response metadata."""
    bucket, artifact_key = _split_s3_uri(artifact_uri)
    prefix = f"{artifact_key}/"

    keys = _list_s3_keys(
        s3_client=s3_client,
        bucket=bucket,
        prefix=prefix,
    )

    raw_directory = f"{prefix}private/raw_responses/"

    raw_response_keys = [
        key for key in keys if key.startswith(raw_directory) and PurePosixPath(key).suffix == ".txt"
    ]

    if len(raw_response_keys) != 1:
        raise RuntimeError(
            "Expected exactly one private raw-response text object under "
            f"{artifact_uri}/private/raw_responses/; "
            f"found {len(raw_response_keys)}"
        )

    metadata_keys = [
        key
        for key in keys
        if key.startswith(raw_directory) and PurePosixPath(key).suffix == ".json"
    ]

    if len(metadata_keys) != 1:
        raise RuntimeError(
            "Expected exactly one private raw-response metadata object under "
            f"{artifact_uri}/private/raw_responses/; "
            f"found {len(metadata_keys)}"
        )

    summary_key = f"{prefix}public_summary.json"

    if summary_key not in keys:
        raise RuntimeError(f"Uncompressed SageMaker model artifact is missing {summary_key!r}")

    raw_key = raw_response_keys[0]

    raw_bytes = _read_s3_bytes(
        s3_client=s3_client,
        bucket=bucket,
        key=raw_key,
    )

    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        raw_response = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Private raw response is not valid UTF-8") from exc

    metadata = _json_object(
        _read_s3_bytes(
            s3_client=s3_client,
            bucket=bucket,
            key=metadata_keys[0],
        ),
        label="raw_response_metadata",
    )

    metadata_sha256 = metadata.get("sha256")

    if not isinstance(metadata_sha256, str):
        raise TypeError("raw_response_metadata.sha256 must be a string")

    if metadata_sha256 != raw_sha256:
        raise RuntimeError(
            "Raw-response metadata digest mismatch: "
            f"computed_sha256={raw_sha256!r}, "
            f"metadata_sha256={metadata_sha256!r}"
        )

    byte_count = metadata.get("byte_count")

    if isinstance(byte_count, bool) or not isinstance(byte_count, int):
        raise TypeError("raw_response_metadata.byte_count must be an integer")

    if byte_count != len(raw_bytes):
        raise RuntimeError(
            "Raw-response metadata byte-count mismatch: "
            f"actual={len(raw_bytes)}, "
            f"metadata={byte_count}"
        )

    response_filename = metadata.get("response_filename")

    if not isinstance(response_filename, str):
        raise TypeError("raw_response_metadata.response_filename must be a string")

    expected_filename = PurePosixPath(raw_key).name

    if response_filename != expected_filename:
        raise RuntimeError(
            "Raw-response metadata filename mismatch: "
            f"actual={expected_filename!r}, "
            f"metadata={response_filename!r}"
        )

    report = _verify_one_raw_response(
        [raw_response],
        model_artifact_uri=artifact_uri,
    )

    artifact_summary = _verify_public_summary(
        _read_s3_bytes(
            s3_client=s3_client,
            bucket=bucket,
            key=summary_key,
        ),
        label="artifact_public_summary",
    )

    return (
        report,
        raw_response,
        raw_sha256,
        artifact_summary,
    )


def _verify_canonical_outputs(
    *,
    s3_client: Any,
    expected_output_s3_prefix: str,
    expected_raw_sha256: str,
    artifact_summary: dict[str, Any],
) -> None:
    """Verify canonical benchmark outputs and their lineage to the raw model response."""
    bucket, output_prefix = _split_s3_uri(expected_output_s3_prefix)

    canonical_summary_key = f"{output_prefix}/public_summary.json"
    private_records_key = f"{output_prefix}/private_records.jsonl"

    canonical_summary = _verify_public_summary(
        _read_s3_bytes(
            s3_client=s3_client,
            bucket=bucket,
            key=canonical_summary_key,
        ),
        label="canonical_public_summary",
    )

    if canonical_summary != artifact_summary:
        raise RuntimeError(
            "Canonical public summary does not exactly match the "
            "SageMaker model-artifact public summary"
        )

    private_records_data = _read_s3_bytes(
        s3_client=s3_client,
        bucket=bucket,
        key=private_records_key,
    )

    try:
        private_records_text = private_records_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Canonical private_records.jsonl is not valid UTF-8") from exc

    lines = [line for line in private_records_text.splitlines() if line.strip()]

    if len(lines) != 1:
        raise RuntimeError(
            "Expected exactly one canonical private record "
            "for the one-question smoke; "
            f"found {len(lines)}"
        )

    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Canonical private record is not valid JSON") from exc

    if not isinstance(record, dict):
        raise TypeError("Canonical private record must be one JSON object")

    prediction = record.get("prediction")

    if not isinstance(prediction, dict):
        raise TypeError("Canonical private record prediction must be a JSON object")

    actual_sha256 = prediction.get("raw_response_sha256")

    if not isinstance(actual_sha256, str):
        raise TypeError("Canonical prediction.raw_response_sha256 must be a string")

    if actual_sha256 != expected_raw_sha256:
        raise RuntimeError(
            "Raw-response lineage mismatch: "
            f"expected_sha256={expected_raw_sha256!r}, "
            f"recorded_sha256={actual_sha256!r}"
        )

    lineage_fields = (
        "experiment_id",
        "protocol_lock_id",
        "asset_manifest_sha256",
        "git_commit_sha",
        "prompt_version",
        "model_key",
        "model_id",
        "model_revision",
    )

    for field in lineage_fields:
        canonical_value = record.get(field)
        summary_value = canonical_summary.get(field)

        if canonical_value != summary_value:
            raise RuntimeError(
                "Canonical record/summary lineage mismatch "
                f"for {field!r}: "
                f"record={canonical_value!r}, "
                f"summary={summary_value!r}"
            )


def verify_training_model_artifact(
    *,
    sagemaker_client: Any,
    s3_client: Any,
    job_name: str,
    expected_output_s3_prefix: str | None = None,
) -> ArtifactGateReport:
    """Verify a completed SageMaker job and its benchmark artifact lineage."""
    description = sagemaker_client.describe_training_job(TrainingJobName=job_name)

    if description.get("TrainingJobStatus") != "Completed":
        raise RuntimeError("Artifact verification requires a Completed SageMaker training job")

    if description.get("HyperParameters", {}).get("mode") == "benchmark":
        return _verify_benchmark_artifact(
            s3_client=s3_client,
            description=description,
            job_name=job_name,
            expected_output_s3_prefix=expected_output_s3_prefix,
        )

    artifact_uri = description.get("ModelArtifacts", {}).get("S3ModelArtifacts")

    if not isinstance(artifact_uri, str) or not artifact_uri.startswith("s3://"):
        raise RuntimeError("Completed training job did not expose an S3 model artifact URI")

    output_config = description.get(
        "OutputDataConfig",
        {},
    )

    if not isinstance(output_config, dict):
        raise TypeError("Completed training job returned malformed OutputDataConfig")

    compression_type = output_config.get("CompressionType")

    if compression_type == "NONE":
        (
            report,
            _raw_response,
            raw_sha256,
            artifact_summary,
        ) = _verify_uncompressed_model_prefix(
            s3_client=s3_client,
            artifact_uri=artifact_uri,
        )

        if expected_output_s3_prefix is not None:
            _verify_canonical_outputs(
                s3_client=s3_client,
                expected_output_s3_prefix=(expected_output_s3_prefix),
                expected_raw_sha256=raw_sha256,
                artifact_summary=artifact_summary,
            )

        return report

    if compression_type in {
        None,
        "",
        "GZIP",
    } and artifact_uri.endswith(
        (
            ".tar.gz",
            ".tgz",
        )
    ):
        bucket, key = _split_s3_uri(artifact_uri)

        data = _read_s3_bytes(
            s3_client=s3_client,
            bucket=bucket,
            key=key,
        )

        return verify_model_artifact_bytes(
            data,
            model_artifact_uri=artifact_uri,
        )

    raise RuntimeError(
        "Unsupported or ambiguous SageMaker "
        "model-artifact layout: "
        f"compression_type={compression_type!r}, "
        f"artifact_uri={artifact_uri!r}"
    )


def inspect_local_model_artifact(path: Path) -> dict[str, object]:
    """Inspect a legacy local model archive without cloud resources."""
    try:
        report = verify_model_artifact_bytes(
            path.read_bytes(),
            model_artifact_uri=str(path),
        )
    except RuntimeError as exc:
        return {
            "verified": False,
            "error": str(exc),
            "artifact": str(path),
        }

    return {
        "verified": True,
        **report.as_dict(),
    }


def _verify_benchmark_artifact(
    *,
    s3_client: Any,
    description: dict[str, Any],
    job_name: str,
    expected_output_s3_prefix: str | None,
) -> ArtifactGateReport:
    """Audit every frozen question, raw generation, parsed result, score, and aggregate."""
    from lava.evaluation.judges import NormalizedExactJudge
    from lava.evaluation.metric import score_question, set_f1
    from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
    from lava.readers.benchmark import _summary
    from lava.readers.evaluation_contract import (
        load_evaluation_contract,
        validate_evaluation_manifest,
    )
    from lava.readers.model_registry import load_resolved_model
    from lava.readers.parsing import ReaderOutputError, parse_reader_response
    from lava.readers.sagemaker_artifacts import canonical_output_s3_prefix
    from lava.readers.schemas import BenchmarkRecord, ReaderPrediction

    root = Path(__file__).resolve().parents[3]
    contract = load_evaluation_contract(root)
    params = description.get("HyperParameters", {})
    if params.get("limit") != str(contract["question_count"]):
        raise RuntimeError("Benchmark job has an unexpected question limit")
    canonical = canonical_output_s3_prefix(description, job_name=job_name)
    if canonical is None or (expected_output_s3_prefix and canonical != expected_output_s3_prefix):
        raise RuntimeError("Benchmark output prefix mismatch")
    bucket, prefix = _split_s3_uri(canonical)
    artifact_uri = description["ModelArtifacts"]["S3ModelArtifacts"]
    if artifact_uri != f"{canonical}/sagemaker-output/{job_name}/output/model":
        raise RuntimeError("Benchmark artifact URI does not belong to this job")
    artifact_bucket, artifact_prefix = _split_s3_uri(artifact_uri)
    if (
        artifact_bucket != bucket
        or description["OutputDataConfig"].get("CompressionType") != "NONE"
    ):
        raise RuntimeError("Benchmark requires uncompressed artifacts in the project bucket")
    model = load_resolved_model(
        root / "configs/oracle_reader_models.lock.json", params["model_key"]
    )
    if contract["models"].get(model.model_key) != description["ResourceConfig"]["InstanceType"]:
        raise RuntimeError("Benchmark hardware differs from the verified model path")
    manifest_bucket, manifest_key = _split_s3_uri(params["manifest_s3_uri"])
    if manifest_bucket != bucket:
        raise RuntimeError("Benchmark manifest bucket mismatch")
    with s3_client.get_object(
        Bucket=bucket, Key=manifest_key, VersionId=contract["private_manifest_version_id"]
    )["Body"] as body:
        examples = validate_evaluation_manifest(body.read(), contract)
    expected = {e.question_id: e for e in examples}

    def read(key: str) -> bytes:
        return _read_s3_bytes(s3_client=s3_client, bucket=bucket, key=key)

    summary = _json_object(read(f"{prefix}/public_summary.json"), label="public_summary")
    artifact_summary = _json_object(
        read(f"{artifact_prefix}/public_summary.json"), label="artifact_summary"
    )
    if summary != artifact_summary:
        raise RuntimeError("Benchmark canonical and artifact summaries differ")
    records = tuple(
        BenchmarkRecord.model_validate_json(line)
        for line in read(f"{prefix}/private_records.jsonl").splitlines()
        if line.strip()
    )
    if len(records) != len(expected) or {r.question_id for r in records} != set(expected):
        raise RuntimeError("Benchmark records have missing, extra, or duplicate questions")
    raw_prefix = f"{artifact_prefix}/private/raw_responses/"
    keys = _list_s3_keys(s3_client=s3_client, bucket=bucket, prefix=raw_prefix)
    metadata_keys = [k for k in keys if k.endswith(".json")]
    text_keys = {k for k in keys if k.endswith(".txt")}
    if len(metadata_keys) != len(expected) or len(text_keys) != len(expected):
        raise RuntimeError("Benchmark raw-response count mismatch")
    raw_by_id: dict[str, str] = {}
    used_keys: set[str] = set()
    for key in metadata_keys:
        metadata = _json_object(read(key), label="raw_metadata")
        question_id = metadata.get("question_id")
        filename = metadata.get("response_filename")
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or question_id not in expected
            or question_id in raw_by_id
        ):
            raise RuntimeError("Benchmark raw metadata has invalid or duplicate identity")
        text_key = raw_prefix + filename
        if text_key not in text_keys or text_key in used_keys:
            raise RuntimeError("Benchmark raw metadata references a missing or reused response")
        raw = _read_s3_bytes(s3_client=s3_client, bucket=bucket, key=text_key, allow_empty=True)
        if hashlib.sha256(raw).hexdigest() != metadata.get("sha256") or len(raw) != metadata.get(
            "byte_count"
        ):
            raise RuntimeError("Benchmark raw-response checksum or byte count mismatch")
        raw_by_id[question_id] = raw.decode("utf-8")
        used_keys.add(text_key)
    lineage = {
        "protocol_lock_id": contract["protocol_lock_id"],
        "asset_manifest_sha256": contract["private_manifest_sha256"],
        "model_key": model.model_key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "experiment_id": params["experiment_id"],
        "git_commit_sha": description["Environment"]["LAVA_GIT_COMMIT_SHA"],
    }
    from lava.readers.prompts import PROMPT_VERSION

    lineage["prompt_version"] = PROMPT_VERSION
    for key, value in lineage.items():
        if summary.get(key) != value or any(getattr(r, key) != value for r in records):
            raise RuntimeError(f"Benchmark lineage mismatch: {key}")
    for record in records:
        example = expected[record.question_id]
        for key in ("document_alias", "language", "answer_format"):
            if getattr(record, key) != getattr(example, key):
                raise RuntimeError(f"Benchmark question alignment mismatch: {key}")
        if record.gold_evidence_pages != example.evidence_pages:
            raise RuntimeError("Benchmark evidence alignment mismatch")
        raw_text = raw_by_id[record.question_id]
        try:
            prediction = parse_reader_response(
                question_id=example.question_id,
                answer_format=example.answer_format,
                raw_response=raw_text,
                allowed_pages=example.evidence_pages,
            )
        except ReaderOutputError as error:
            prediction = ReaderPrediction(
                question_id=example.question_id,
                answer_format=example.answer_format,
                answer="",
                evidence_pages=(),
                confidence=0.0,
                abstain=True,
                schema_valid=False,
                parser_error_code=error.code,
                raw_response_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
            )
        if record.prediction != prediction:
            raise RuntimeError("Benchmark parsed prediction differs from raw generation")
        reference = ReferenceRecord(
            question_id=example.question_id,
            document_id=example.document_id,
            question=example.question,
            answer_format=example.answer_format,
            answer=example.answer,
            evidence_pages=example.evidence_pages,
            language=example.language,
        )
        score = score_question(
            reference,
            PredictionRecord(
                question_id=example.question_id,
                answer=prediction.answer,
                evidence_pages=example.evidence_pages,
            ),
            judge=NormalizedExactJudge(),
        )
        if (
            record.normalized_exact_answer_score != score.answer_score
            or record.self_grounding_f1 != set_f1(example.evidence_pages, prediction.evidence_pages)
            or record.oracle_fixed_overall_diagnostic != (score.answer_score + 1.0) / 2.0
        ):
            raise RuntimeError("Benchmark score differs from independently recomputed score")
    if (
        summary.get("run_kind") != "benchmark"
        or summary.get("coverage_status") != "complete_frozen_pilot"
    ):
        raise RuntimeError("Benchmark summary lacks explicit complete-pilot status")
    for key, value in _summary(records).items():
        if summary.get(key) != value:
            raise RuntimeError(f"Benchmark aggregate mismatch: {key}")
    for key, value in {
        "generation": model.generation.model_dump(mode="json"),
        "input_mode": model.input_mode.value,
        "dtype": model.dtype,
        "quantization": model.quantization.value,
    }.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Benchmark model contract mismatch: {key}")
    return ArtifactGateReport(
        raw_response_count=len(records),
        schema_valid_rate=summary["schema_valid_rate"],
        parser_error_counts={
            k: v for k, v in summary["parser_error_counts"].items() if k != "none"
        },
        model_artifact_uri=artifact_uri,
    )
