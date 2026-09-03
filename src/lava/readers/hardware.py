"""Pinned SageMaker GPU hardware contracts.

Values are frozen from the AWS EC2 accelerated-instance specification
table checked on 2026-09-03. Unknown hardware fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from lava.readers.schemas import DevicePlacement

HARDWARE_CATALOG_VERSION: Final = "aws-ec2-accelerated-2026-09-03"
HARDWARE_CATALOG_SOURCE: Final = "https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html"


@dataclass(frozen=True, slots=True)
class GpuHardwareSpec:
    """Immutable accelerator capacity for one SageMaker instance."""

    instance_type: str
    gpu_model: str
    gpu_count: int
    gpu_memory_per_device_gib: int
    total_gpu_memory_gib: int


class HardwareContract(Protocol):
    """Minimum model resource requirements used for fit validation."""

    instance_type: str
    device_placement: DevicePlacement
    min_cuda_devices: int
    min_cuda_memory_per_device_gib: int
    min_total_cuda_memory_gib: int


_GPU_HARDWARE: Final[Mapping[str, GpuHardwareSpec]] = MappingProxyType(
    {
        "ml.g5.2xlarge": GpuHardwareSpec(
            instance_type="ml.g5.2xlarge",
            gpu_model="NVIDIA A10G",
            gpu_count=1,
            gpu_memory_per_device_gib=22,
            total_gpu_memory_gib=22,
        ),
        "ml.g6e.2xlarge": GpuHardwareSpec(
            instance_type="ml.g6e.2xlarge",
            gpu_model="NVIDIA L40S",
            gpu_count=1,
            gpu_memory_per_device_gib=44,
            total_gpu_memory_gib=44,
        ),
        "ml.g7e.12xlarge": GpuHardwareSpec(
            instance_type="ml.g7e.12xlarge",
            gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            gpu_count=2,
            gpu_memory_per_device_gib=96,
            total_gpu_memory_gib=192,
        ),
        "ml.g7e.24xlarge": GpuHardwareSpec(
            instance_type="ml.g7e.24xlarge",
            gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            gpu_count=4,
            gpu_memory_per_device_gib=96,
            total_gpu_memory_gib=384,
        ),
        "ml.g7e.48xlarge": GpuHardwareSpec(
            instance_type="ml.g7e.48xlarge",
            gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            gpu_count=8,
            gpu_memory_per_device_gib=96,
            total_gpu_memory_gib=768,
        ),
        "ml.p5en.48xlarge": GpuHardwareSpec(
            instance_type="ml.p5en.48xlarge",
            gpu_model="NVIDIA H200",
            gpu_count=8,
            gpu_memory_per_device_gib=141,
            total_gpu_memory_gib=1128,
        ),
        "ml.p6-b200.48xlarge": GpuHardwareSpec(
            instance_type="ml.p6-b200.48xlarge",
            gpu_model="NVIDIA B200",
            gpu_count=8,
            gpu_memory_per_device_gib=179,
            total_gpu_memory_gib=1432,
        ),
        "ml.p6-b300.48xlarge": GpuHardwareSpec(
            instance_type="ml.p6-b300.48xlarge",
            gpu_model="NVIDIA B300 Blackwell Ultra",
            gpu_count=8,
            gpu_memory_per_device_gib=268,
            total_gpu_memory_gib=2148,
        ),
    }
)


def get_gpu_hardware(instance_type: str) -> GpuHardwareSpec:
    """Return pinned accelerator capacity or reject unsupported hardware."""
    try:
        return _GPU_HARDWARE[instance_type]
    except KeyError as error:
        raise ValueError(
            f"SageMaker GPU instance is absent from the pinned hardware catalog: {instance_type!r}"
        ) from error


def validate_hardware_fit(
    contract: HardwareContract,
) -> GpuHardwareSpec:
    """Fail closed when a model contract exceeds instance capacity."""
    hardware = get_gpu_hardware(contract.instance_type)

    if hardware.gpu_count < contract.min_cuda_devices:
        raise ValueError(
            "Insufficient GPU count: "
            f"available={hardware.gpu_count}, "
            f"required={contract.min_cuda_devices}"
        )

    if hardware.gpu_memory_per_device_gib < contract.min_cuda_memory_per_device_gib:
        raise ValueError(
            "Insufficient per-GPU memory: "
            f"available_gib={hardware.gpu_memory_per_device_gib}, "
            f"required_gib={contract.min_cuda_memory_per_device_gib}"
        )

    if hardware.total_gpu_memory_gib < contract.min_total_cuda_memory_gib:
        raise ValueError(
            "Insufficient aggregate GPU memory: "
            f"available_gib={hardware.total_gpu_memory_gib}, "
            f"required_gib={contract.min_total_cuda_memory_gib}"
        )

    if (
        contract.device_placement is DevicePlacement.SINGLE
        and contract.min_total_cuda_memory_gib > hardware.gpu_memory_per_device_gib
    ):
        raise ValueError("Single-device placement cannot borrow aggregate memory from other GPUs")

    return hardware
