import pytest

from lava.readers.schemas import (
    DecodingMode,
    GenerationSpec,
    ReaderInputMode,
    SageMakerJobPlan,
)


def test_deterministic_generation_rejects_sampling_parameters() -> None:
    with pytest.raises(ValueError):
        GenerationSpec(
            mode=DecodingMode.DIRECT,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.7,
            seed=1,
        )


def test_thinking_requires_sampled_decoding() -> None:
    with pytest.raises(ValueError, match="thinking mode"):
        GenerationSpec(
            mode=DecodingMode.THINKING,
            max_new_tokens=64,
            do_sample=False,
            seed=1,
        )


def test_sagemaker_plan_cannot_create_multiple_instances() -> None:
    generation = GenerationSpec(
        mode=DecodingMode.DIRECT,
        max_new_tokens=64,
        do_sample=False,
        seed=1,
    )
    with pytest.raises(ValueError):
        SageMakerJobPlan(
            sdk_version="3.21.0",
            model_key="m",
            model_id="org/model",
            model_revision="a" * 40,
            protocol_lock_id="b" * 64,
            git_commit_sha="c" * 40,
            bucket="bucket",
            manifest_s3_uri="s3://bucket/manifest.jsonl",
            output_s3_prefix="s3://bucket/results",
            training_image="image",
            training_image_digest="sha256:" + "d" * 64,
            instance_type="ml.g5.2xlarge",
            instance_count=2,
            volume_size_gb=100,
            max_runtime_seconds=3600,
            max_wait_seconds=3600,
            managed_spot=False,
            limit=1,
            input_mode=ReaderInputMode.FUSED,
            generation=generation,
        )
