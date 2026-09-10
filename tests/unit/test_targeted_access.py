"""Prevent administrator-only preflight checks and version-read permission regressions."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "targeted_access_probe", ROOT / "scripts/verify_targeted_access.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_policy_allows_pinned_version_reads_without_widening_scope():
    policy = json.loads((ROOT / "infra/iam/submission.template.json").read_bytes())
    assert policy["Statement"] == [
        {
            "Sid": "ReadWriteSubmissionArtifacts",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
            "Resource": "arn:aws:s3:::${S3_BUCKET}/experiments/submissions/*",
        }
    ]


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:iam::123456789012:user/admin",
        "arn:aws:sts::123456789012:assumed-role/wrong/SageMaker",
        "arn:aws:sts::999999999999:assumed-role/reader/SageMaker",
    ],
)
def test_wrong_principal_fails_before_reading_or_writing_any_artifact(probe, arn):
    s3, sts = Mock(), Mock()
    sts.get_caller_identity.return_value = {"Account": "123456789012", "Arn": arn}
    with pytest.raises(ValueError, match="exact SageMaker role"):
        probe.run(
            ROOT,
            s3,
            sts,
            {"LAVA_EXPECTED_ROLE_ARN": "arn:aws:iam::123456789012:role/service-role/reader"},
        )
    assert not s3.mock_calls


def test_exact_assumed_execution_role_is_accepted(probe):
    probe.verify_caller(
        {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/reader/SageMaker",
        },
        "arn:aws:iam::123456789012:role/service-role/reader",
    )


def test_unversioned_asset_is_checked_again_at_observed_version(probe):
    s3 = Mock()
    s3.get_object.side_effect = [
        {"Body": io.BytesIO(b"image"), "VersionId": "observed"},
        {"Body": io.BytesIO(b"image"), "VersionId": "observed"},
    ]
    probe.verify_asset(
        s3,
        "bucket",
        {
            "image_s3_uri": "s3://bucket/image.png",
            "image_sha256": hashlib.sha256(b"image").hexdigest(),
            "image_version_id": None,
        },
        "image",
    )
    assert s3.get_object.call_args.kwargs == {
        "Bucket": "bucket",
        "Key": "image.png",
        "VersionId": "observed",
    }


def test_asset_version_denial_is_not_hidden(probe):
    s3 = Mock()
    s3.get_object.side_effect = PermissionError("GetObjectVersion denied")
    with pytest.raises(PermissionError, match="GetObjectVersion"):
        probe.verify_asset(
            s3,
            "bucket",
            {
                "image_s3_uri": "s3://bucket/image.png",
                "image_sha256": hashlib.sha256(b"image").hexdigest(),
                "image_version_id": "pinned",
            },
            "image",
        )
