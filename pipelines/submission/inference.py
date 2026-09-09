"""Run the frozen test reader with one visible GPU and a 36 GiB allocator limit."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
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


def export_outputs(root: Path, s3: Any, bucket: str, identity: str, logger: Any) -> dict[str, Any]:
    """Persist the validated CSV and a stable receipt after complete inference."""
    from lava.readers.system import store_for
    from lava.readers.test_inference import export_test

    target = export_test(root, s3, bucket, logger)
    manifest = json.loads(target.with_name("manifest.json").read_bytes())
    result = {
        "bundle_id": target.parent.name,
        "submission_sha256": manifest["submission_sha256"],
        "csv_key": f"experiments/submissions/lava-challenge-2026/{target.parent.name}/submission.csv",
        "validation": manifest["validation"],
        "uploaded_to_kaggle": False,
    }
    store = store_for(s3, bucket, identity)
    store.write("export.json", result)
    if store.read("export.json") != result:
        raise ValueError("Submission export receipt failed durable read-back")
    return result


def main() -> int:
    import boto3
    import torch

    from lava.readers.runtime_logging import RuntimeEventLogger

    print(json.dumps(configure_runtime(torch), sort_keys=True), flush=True)
    run_inference()
    logger = RuntimeEventLogger(
        "submission.export", jsonl_path=Path("/opt/ml/model/private/events.jsonl")
    )
    with logger.stage("submission.export", heartbeat_seconds=15):
        export_outputs(
            Path(__file__).resolve().parents[2],
            boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"]),
            os.environ["LAVA_BUCKET"],
            os.environ["LAVA_SYSTEM_CONTRACT"],
            logger,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
