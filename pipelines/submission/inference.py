"""Run the frozen test reader with one visible GPU and a 36 GiB allocator limit."""

from __future__ import annotations

import json
import runpy
from typing import Any


def configure_runtime(torch_module: Any) -> dict[str, Any]:
    """Check the device boundary before loading model weights."""
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError("Submission inference requires exactly one visible CUDA device")
    properties = torch_module.cuda.get_device_properties(0)
    limit_bytes = 36 * 1024**3
    if properties.total_memory < 38 * 1024**3:
        raise RuntimeError("The selected device lacks the required bounded memory headroom")
    torch_module.cuda.set_device(0)
    torch_module.cuda.set_per_process_memory_fraction(limit_bytes / properties.total_memory, 0)
    return {
        "event": "submission.gpu.boundary.verified",
        "visible_cuda_devices": 1,
        "gpu_name": properties.name,
        "allocator_limit_gib": 36,
        "reference_memory_limit_gib": 40,
    }


def run_inference() -> None:
    """Register __main__ so Pydantic can resolve the reader's forward annotations."""
    runpy.run_module("lava.readers.test_inference", run_name="__main__", alter_sys=True)


def main() -> int:
    import torch

    print(json.dumps(configure_runtime(torch), sort_keys=True), flush=True)
    run_inference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
