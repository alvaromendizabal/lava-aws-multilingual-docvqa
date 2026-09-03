from __future__ import annotations

import pytest

from lava.readers.hardware import (
    HARDWARE_CATALOG_VERSION,
    get_gpu_hardware,
    validate_hardware_fit,
)
from lava.readers.schemas import (
    DecodingMode,
    DevicePlacement,
    GenerationSpec,
    ModelCandidate,
    ReaderInputMode,
)


def _candidate(
    *,
    instance_type: str,
    placement: DevicePlacement = DevicePlacement.SINGLE,
    min_devices: int = 1,
    min_per_device_gib: int = 1,
    min_total_gib: int = 1,
) -> ModelCandidate:
    return ModelCandidate(
        model_key="candidate",
        model_id="Org/Model",
        expected_license="apache-2.0",
        expected_pipeline_tag="image-text-to-text",
        parameters_billion=100.0,
        device_placement=placement,
        min_cuda_devices=min_devices,
        min_cuda_memory_per_device_gib=min_per_device_gib,
        min_total_cuda_memory_gib=min_total_gib,
        instance_type=instance_type,
        input_mode=ReaderInputMode.FUSED,
        dtype="bfloat16",
        attention_implementation="sdpa",
        use_kernels=False,
        processor_min_pixels=200704,
        processor_max_pixels=1605632,
        generation=GenerationSpec(
            mode=DecodingMode.DIRECT,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.0,
            seed=1,
        ),
    )


def test_hardware_catalog_is_versioned() -> None:
    assert HARDWARE_CATALOG_VERSION == "aws-ec2-accelerated-2026-09-03"


@pytest.mark.parametrize(
    ("instance_type", "gpu_count", "per_device", "total"),
    [
        ("ml.g5.2xlarge", 1, 22, 22),
        ("ml.g6e.2xlarge", 1, 44, 44),
        ("ml.g7e.12xlarge", 2, 96, 192),
        ("ml.g7e.48xlarge", 8, 96, 768),
        ("ml.p5en.48xlarge", 8, 141, 1128),
        ("ml.p6-b200.48xlarge", 8, 179, 1432),
        ("ml.p6-b300.48xlarge", 8, 268, 2148),
    ],
)
def test_pinned_aws_gpu_capacity(
    instance_type: str,
    gpu_count: int,
    per_device: int,
    total: int,
) -> None:
    hardware = get_gpu_hardware(instance_type)

    assert hardware.gpu_count == gpu_count
    assert hardware.gpu_memory_per_device_gib == per_device
    assert hardware.total_gpu_memory_gib == total


def test_p6_hyphenated_instance_name_is_valid() -> None:
    candidate = _candidate(
        instance_type="ml.p6-b300.48xlarge",
        placement=DevicePlacement.AUTO_SHARDED,
        min_devices=8,
    )

    assert candidate.instance_type == "ml.p6-b300.48xlarge"


def test_p6_b300_large_contract_passes() -> None:
    candidate = _candidate(
        instance_type="ml.p6-b300.48xlarge",
        placement=DevicePlacement.AUTO_SHARDED,
        min_devices=8,
        min_per_device_gib=200,
        min_total_gib=1800,
    )

    hardware = validate_hardware_fit(candidate)

    assert hardware.total_gpu_memory_gib == 2148


def test_insufficient_gpu_count_fails_closed() -> None:
    candidate = _candidate(
        instance_type="ml.g7e.12xlarge",
        placement=DevicePlacement.AUTO_SHARDED,
        min_devices=4,
    )

    with pytest.raises(ValueError, match="GPU count"):
        validate_hardware_fit(candidate)


def test_insufficient_per_gpu_memory_fails_closed() -> None:
    candidate = _candidate(
        instance_type="ml.g7e.48xlarge",
        placement=DevicePlacement.AUTO_SHARDED,
        min_devices=8,
        min_per_device_gib=128,
    )

    with pytest.raises(ValueError, match="per-GPU memory"):
        validate_hardware_fit(candidate)


def test_insufficient_total_memory_fails_closed() -> None:
    candidate = _candidate(
        instance_type="ml.g7e.48xlarge",
        placement=DevicePlacement.AUTO_SHARDED,
        min_devices=8,
        min_total_gib=1000,
    )

    with pytest.raises(ValueError, match="aggregate GPU memory"):
        validate_hardware_fit(candidate)


def test_single_gpu_placement_cannot_borrow_other_gpu_memory() -> None:
    candidate = _candidate(
        instance_type="ml.p6-b300.48xlarge",
        min_total_gib=300,
    )

    with pytest.raises(ValueError, match="cannot borrow"):
        validate_hardware_fit(candidate)


def test_unknown_gpu_instance_fails_closed() -> None:
    candidate = _candidate(
        instance_type="ml.unknown-gpu.1xlarge",
    )

    with pytest.raises(ValueError, match="absent from the pinned"):
        validate_hardware_fit(candidate)
