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
