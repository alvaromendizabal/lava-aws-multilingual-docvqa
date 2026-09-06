from __future__ import annotations

from lava.readers.sagemaker_artifacts import (
    canonical_output_s3_prefix,
    model_artifact_uri,
    split_s3_uri,
)


def _description(*, compression_type: str = "NONE") -> dict[str, object]:
    return {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {
            "S3ModelArtifacts": (
                "s3://bucket/experiments/run/sagemaker-output/lava-oracle-job/output/model"
            )
        },
        "OutputDataConfig": {
            "CompressionType": compression_type,
            "S3OutputPath": "s3://bucket/experiments/run/sagemaker-output",
        },
    }


def test_model_artifact_uri_reads_training_job_artifact() -> None:
    assert model_artifact_uri(_description()).endswith("/lava-oracle-job/output/model")


def test_uncompressed_output_prefix_is_derived_from_output_config() -> None:
    actual = canonical_output_s3_prefix(
        _description(),
        job_name="lava-oracle-job",
    )

    assert actual == "s3://bucket/experiments/run"


def test_uncompressed_output_prefix_falls_back_to_artifact_uri() -> None:
    description = _description()
    description["OutputDataConfig"] = {"CompressionType": "NONE"}

    actual = canonical_output_s3_prefix(
        description,
        job_name="lava-oracle-job",
    )

    assert actual == "s3://bucket/experiments/run"


def test_split_s3_uri() -> None:
    assert split_s3_uri("s3://bucket/path/to/object") == (
        "bucket",
        "path/to/object",
    )
