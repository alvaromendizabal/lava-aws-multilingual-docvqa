"""Tests for protocol construction, privacy boundaries, and S3 verification."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from lava.evaluation import protocol

_TRAIN = b"""id,file_id,question,answer_format,answer,evidence_page_number,language
q1,j_1,question one,string,answer one,[1],ja
q2,j_2,question two,number,2,[2],ja
q3,v_1,question three,unordered_list,"['a','b']","[1,2]",vi
"""


class FakeS3:
    """Minimal in-memory S3 double covering protocol I/O and metadata checks."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        key = str(kwargs["Key"])
        if key != "raw/kaggle/train.csv":
            raise AssertionError(key)
        return {"Body": io.BytesIO(_TRAIN)}

    def head_object(self, **kwargs: object) -> dict[str, Any]:
        key = str(kwargs["Key"])
        if key.startswith("raw/kaggle/train_pdfs/"):
            return {"ContentLength": 100}
        payload, metadata = self.objects[key]
        return {"ContentLength": len(payload), "Metadata": metadata}

    def put_object(self, **kwargs: object) -> dict[str, str]:
        key = str(kwargs["Key"])
        body = kwargs["Body"]
        metadata = kwargs["Metadata"]
        if not isinstance(body, bytes) or not isinstance(metadata, dict):
            raise TypeError("Fake S3 requires bytes and metadata")
        self.objects[key] = (body, {str(k): str(v) for k, v in metadata.items()})
        return {"VersionId": f"version-{len(self.objects)}"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_freeze_protocol_is_nested_private_and_provenance_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protocol must create nested folds without leaking private labels publicly."""
    monkeypatch.chdir(tmp_path)
    fake_s3 = FakeS3()
    monkeypatch.setattr(protocol.boto3, "client", lambda *_args, **_kwargs: fake_s3)
    monkeypatch.setattr(protocol, "_git_head", lambda: "deadbeef")

    code_paths: list[str] = []
    for name in (
        "schemas.py",
        "normalization.py",
        "judges.py",
        "matching.py",
        "metric.py",
        "retrieval.py",
        "splits.py",
        "statistics.py",
        "protocol.py",
    ):
        path = Path("src/lava/evaluation") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        code_paths.append(str(path))
    monkeypatch.setattr(protocol, "_CODE_PATHS", tuple(code_paths))

    config = {
        "protocol_name": "test-protocol",
        "expected_question_count": 3,
        "expected_document_count": 3,
        "random_seed": 7,
        "official_metric_spec": {},
        "retrieval_metric_spec": {},
        "selection_policy": {},
        "competition_constraints": {},
        "judge_specification_boundary": {},
    }
    audit = {
        "pdf_success_count": 205,
        "pdf_error_count": 0,
        "audit_source_sha256": "audit-source",
        "csv_profiles": [
            {
                "s3_key": "raw/kaggle/train.csv",
                "selected_columns": {"answer": "answer"},
            },
            {
                "s3_key": "raw/kaggle/test.csv",
                "selected_columns": {"answer": None},
            },
        ],
    }
    _write_json(Path("configs/evaluation_protocol.json"), config)
    _write_json(Path("reports/data_audit/data_audit_summary_full.json"), audit)

    summary = protocol.freeze_protocol(
        bucket="private-bucket",
        region="us-west-2",
        config_path=Path("configs/evaluation_protocol.json"),
        audit_path=Path("reports/data_audit/data_audit_summary_full.json"),
    )

    assert summary["question_count"] == 3
    assert summary["document_count"] == 3
    assert summary["outer_fold_count"] == 3
    assert summary["inner_fold_count_total"] == 6
    assert len(fake_s3.objects) == 6
    public_text = json.dumps(summary, ensure_ascii=False)
    assert "question one" not in public_text
    assert "answer one" not in public_text
    assert '"j_1"' not in public_text
    assert Path("artifacts/evaluation_protocol/reference_records.jsonl").exists()
    assert Path("configs/evaluation_protocol.lock.json").exists()
    assert Path("reports/evaluation/EVALUATION_PROTOCOL.md").exists()
