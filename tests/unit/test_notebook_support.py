"""Tests for notebook environment and public-metadata helpers."""

from __future__ import annotations

from lava.notebook_support import find_repo_root, public_metadata


def test_find_repo_root_from_nested_directory(tmp_path) -> None:
    """Repository discovery must work from a notebook-like child directory."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = tmp_path / "notebooks" / "nested"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path


def test_public_metadata_drops_private_locations() -> None:
    """Public displays must omit bucket and private S3 fields."""
    result = public_metadata(
        {
            "bucket": "private",
            "manifest_s3_uri": "s3://private/path",
            "model_key": "model",
            "aws_arn": "arn:aws:iam::123456789012:user/name",
        }
    )
    assert "bucket" not in result
    assert "manifest_s3_uri" not in result
    assert result["model_key"] == "model"
    assert result["aws_arn"] == "arn:aws:iam::<redacted-account>:user/name"
