from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_script() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "sync_oracle_reader_results.py"
    spec = importlib.util.spec_from_file_location(
        "sync_oracle_reader_results_under_test",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _description() -> dict[str, Any]:
    return {
        "TrainingJobStatus": "Completed",
        "TrainingTimeInSeconds": 350,
        "BillableTimeInSeconds": 350,
        "ModelArtifacts": {
            "S3ModelArtifacts": (
                "s3://bucket/experiments/run/sagemaker-output/lava-oracle-job/output/model"
            )
        },
        "OutputDataConfig": {
            "CompressionType": "NONE",
            "S3OutputPath": "s3://bucket/experiments/run/sagemaker-output",
        },
        "ResourceConfig": {"InstanceType": "ml.g6e.2xlarge"},
    }


def test_job_timing_preserves_utc_and_separates_capacity_wait() -> None:
    module = _load_script()
    description = {
        "CreationTime": datetime(2026, 9, 6, 19, 0, tzinfo=UTC),
        "TrainingStartTime": "2026-09-06T12:02:00-07:00",
        "TrainingEndTime": "2026-09-06T19:05:30Z",
        "SecondaryStatusTransitions": [
            {
                "Status": "Pending",
                "StartTime": "2026-09-06T19:00:01Z",
                "EndTime": "2026-09-06T19:01:00Z",
            },
            {
                "Status": "Pending",
                "StartTime": "2026-09-06T19:01:00Z",
                "EndTime": "2026-09-06T19:02:00Z",
            },
            {
                "Status": "Training",
                "StartTime": "2026-09-06T19:02:00Z",
                "EndTime": "2026-09-06T19:05:30Z",
            },
        ],
    }
    result = module._job_timing(description)
    assert result["created_at_utc"] == "2026-09-06T19:00:00+00:00"
    assert result["compute_started_at_utc"] == "2026-09-06T19:02:00+00:00"
    assert result["elapsed_seconds"] == 330
    assert result["phase_seconds"] == {"Pending": 119, "Training": 210}


def test_missing_job_timing_remains_unknown() -> None:
    result = _load_script()._job_timing({})
    assert result["elapsed_seconds"] is None
    assert result["phase_seconds"] == {}
    assert result["compute_started_at_utc"] is None


@pytest.mark.parametrize(
    "description",
    [
        {"CreationTime": "2026-09-06T19:00:00"},
        {"CreationTime": "2026-09-06T19:00:00Z", "TrainingEndTime": "2026-09-06T18:00:00Z"},
        {
            "SecondaryStatusTransitions": [
                {
                    "Status": "Pending",
                    "StartTime": "2026-09-06T19:00:00Z",
                    "EndTime": "2026-09-06T18:00:00Z",
                }
            ]
        },
    ],
)
def test_job_timing_rejects_ambiguous_or_reversed_timestamps(description) -> None:
    with pytest.raises(ValueError):
        _load_script()._job_timing(description)


def test_sync_uses_job_metadata_not_shell_bucket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    summary = {
        "model_key": "qwen35_9b_fused_direct",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "revision",
        "git_commit_sha": "abcdef",
        "protocol_lock_id": "lock",
        "record_count": 1,
        "schema_valid_rate": 1.0,
    }
    payload = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    class FakeSageMaker:
        def describe_training_job(self, **_: object) -> dict[str, Any]:
            return _description()

    class FakeS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {
                "Bucket": "bucket",
                "Key": "experiments/run/public_summary.json",
            }
            return {
                "Body": io.BytesIO(payload),
                "Metadata": {"sha256": digest},
            }

    fake_sagemaker = FakeSageMaker()
    fake_s3 = FakeS3()

    class FakeSession:
        def client(self, name: str) -> object:
            if name == "sagemaker":
                return fake_sagemaker
            if name == "s3":
                return fake_s3
            raise AssertionError(name)

    monkeypatch.setattr(module, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(
        module.boto3.session,
        "Session",
        lambda region_name: FakeSession(),
    )
    monkeypatch.setattr(
        module,
        "verify_training_model_artifact",
        lambda **_: SimpleNamespace(
            as_dict=lambda: {
                "raw_response_count": 1,
                "schema_valid_rate": 1.0,
                "parser_error_counts": {},
                "model_artifact_uri": "s3://bucket/model",
            }
        ),
    )
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_oracle_reader_results.py",
            "--job-name",
            "lava-oracle-job",
        ],
    )

    assert module.main() == 0

    run_summary = tmp_path / "reports/oracle_reader/runs/lava-oracle-job/public_summary.json"
    latest_summary = tmp_path / "reports/oracle_reader/latest_public_summary.json"
    manifest = tmp_path / "reports/oracle_reader/runs/lava-oracle-job/sync_manifest.json"

    assert run_summary.read_bytes() == payload
    assert latest_summary.read_bytes() == payload
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["instance_type"] == "ml.g6e.2xlarge"
    assert manifest_payload["public_summary_sha256"] == digest

    output = capsys.readouterr().out
    assert "results.sync.artifact_verified" in output
    assert "ORACLE_READER_RESULTS_SYNCED" in output
