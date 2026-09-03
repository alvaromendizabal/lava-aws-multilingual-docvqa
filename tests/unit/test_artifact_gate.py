from __future__ import annotations

import io
import tarfile

import pytest

from lava.readers.artifact_gate import verify_model_artifact_bytes


def _archive(raw_responses: list[str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for index, raw in enumerate(raw_responses, start=1):
            encoded = raw.encode("utf-8")
            info = tarfile.TarInfo(f"private/raw_responses/response-{index:06d}.txt")
            info.size = len(encoded)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(encoded))
    return buffer.getvalue()


def test_one_schema_valid_response_passes() -> None:
    raw = '{"answer":"yes","evidence_pages":[1],"confidence":0.9,"abstain":false}'
    report = verify_model_artifact_bytes(_archive([raw]))
    assert report.raw_response_count == 1
    assert report.schema_valid_rate == 1.0
    assert report.parser_error_counts == {}


def test_zero_raw_responses_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_model_artifact_bytes(_archive([]))


def test_multiple_raw_responses_fail_for_one_question_smoke() -> None:
    raw = '{"answer":"yes","evidence_pages":[1],"confidence":0.9,"abstain":false}'
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_model_artifact_bytes(_archive([raw, raw]))


def test_invalid_schema_fails_even_when_sagemaker_would_be_complete() -> None:
    with pytest.raises(RuntimeError, match="structured-output gate"):
        verify_model_artifact_bytes(_archive(["not json"]))


_PHASE5E_ARTIFACT_PREFIX = "experiments/run/sagemaker-output/job/output/model"

_PHASE5E_OUTPUT_PREFIX = "experiments/run"

_PHASE5E_LINEAGE = {
    "experiment_id": "experiment",
    "protocol_lock_id": "p" * 64,
    "asset_manifest_sha256": "a" * 64,
    "git_commit_sha": "b" * 40,
    "prompt_version": "oracle-reader-json-v3",
    "model_key": "model-key",
    "model_id": "org/model",
    "model_revision": "c" * 40,
}


def _phase5e_summary() -> dict[str, object]:
    return {
        **_PHASE5E_LINEAGE,
        "record_count": 1,
        "schema_valid_rate": 1.0,
        "abstention_rate": 0.0,
        "parser_error_counts": {
            "none": 1,
        },
    }


def _phase5e_objects(
    raw: str,
    *,
    metadata_sha256: str | None = None,
    canonical_sha256: str | None = None,
    include_raw: bool = True,
) -> dict[str, bytes]:
    import hashlib
    import json

    raw_bytes = raw.encode("utf-8")
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    metadata_sha256 = raw_sha256 if metadata_sha256 is None else metadata_sha256

    canonical_sha256 = raw_sha256 if canonical_sha256 is None else canonical_sha256

    summary_bytes = json.dumps(
        _phase5e_summary(),
        sort_keys=True,
    ).encode("utf-8")

    objects: dict[str, bytes] = {
        (f"{_PHASE5E_ARTIFACT_PREFIX}/public_summary.json"): summary_bytes,
        (f"{_PHASE5E_OUTPUT_PREFIX}/public_summary.json"): summary_bytes,
        (f"{_PHASE5E_OUTPUT_PREFIX}/private_records.jsonl"): (
            json.dumps(
                {
                    **_PHASE5E_LINEAGE,
                    "prediction": {"raw_response_sha256": (canonical_sha256)},
                },
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        ),
    }

    if include_raw:
        raw_key = f"{_PHASE5E_ARTIFACT_PREFIX}/private/raw_responses/response-000001.txt"

        metadata_key = f"{_PHASE5E_ARTIFACT_PREFIX}/private/raw_responses/response-000001.json"

        objects[raw_key] = raw_bytes

        objects[metadata_key] = json.dumps(
            {
                "byte_count": len(raw_bytes),
                "response_filename": ("response-000001.txt"),
                "schema_version": 1,
                "sequence": 1,
                "sha256": metadata_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")

    return objects


class _Phase5ESageMakerClient:
    def describe_training_job(
        self,
        **_: object,
    ) -> dict[str, object]:
        return {
            "TrainingJobStatus": "Completed",
            "ModelArtifacts": {"S3ModelArtifacts": (f"s3://bucket/{_PHASE5E_ARTIFACT_PREFIX}")},
            "OutputDataConfig": {
                "CompressionType": "NONE",
            },
        }


class _Phase5ES3Client:
    def __init__(
        self,
        objects: dict[str, bytes],
    ) -> None:
        self.objects = objects

    def list_objects_v2(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        prefix = str(kwargs["Prefix"])

        return {
            "IsTruncated": False,
            "Contents": [
                {
                    "Key": key,
                }
                for key in sorted(self.objects)
                if key.startswith(prefix)
            ],
        }

    def get_object(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        import io

        key = str(kwargs["Key"])

        return {
            "Body": io.BytesIO(self.objects[key]),
        }


def _phase5e_raw() -> str:
    return '{"answer":"x","evidence_pages":[1],"confidence":0.9,"abstain":false}'


def test_uncompressed_sagemaker_artifact_prefix_is_verified() -> None:
    from lava.readers.artifact_gate import (
        verify_training_model_artifact,
    )

    report = verify_training_model_artifact(
        sagemaker_client=(_Phase5ESageMakerClient()),
        s3_client=_Phase5ES3Client(_phase5e_objects(_phase5e_raw())),
        job_name="job",
        expected_output_s3_prefix=(f"s3://bucket/{_PHASE5E_OUTPUT_PREFIX}"),
    )

    assert report.raw_response_count == 1
    assert report.schema_valid_rate == 1.0
    assert report.parser_error_counts == {}
    assert report.model_artifact_uri == (f"s3://bucket/{_PHASE5E_ARTIFACT_PREFIX}")


def test_uncompressed_artifact_missing_raw_response_fails_closed() -> None:
    import pytest

    from lava.readers.artifact_gate import (
        verify_training_model_artifact,
    )

    objects = _phase5e_objects(
        _phase5e_raw(),
        include_raw=False,
    )

    with pytest.raises(
        RuntimeError,
        match="exactly one private raw-response",
    ):
        verify_training_model_artifact(
            sagemaker_client=(_Phase5ESageMakerClient()),
            s3_client=_Phase5ES3Client(objects),
            job_name="job",
        )


def test_raw_response_metadata_digest_mismatch_fails_closed() -> None:
    import pytest

    from lava.readers.artifact_gate import (
        verify_training_model_artifact,
    )

    objects = _phase5e_objects(
        _phase5e_raw(),
        metadata_sha256="0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match=("Raw-response metadata digest mismatch"),
    ):
        verify_training_model_artifact(
            sagemaker_client=(_Phase5ESageMakerClient()),
            s3_client=_Phase5ES3Client(objects),
            job_name="job",
        )


def test_raw_response_lineage_digest_mismatch_fails_closed() -> None:
    import pytest

    from lava.readers.artifact_gate import (
        verify_training_model_artifact,
    )

    objects = _phase5e_objects(
        _phase5e_raw(),
        canonical_sha256="0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="Raw-response lineage mismatch",
    ):
        verify_training_model_artifact(
            sagemaker_client=(_Phase5ESageMakerClient()),
            s3_client=_Phase5ES3Client(objects),
            job_name="job",
            expected_output_s3_prefix=(f"s3://bucket/{_PHASE5E_OUTPUT_PREFIX}"),
        )
