"""Validate the installed SageMaker SDK V3 contract without creating resources."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from lava.readers.sagemaker import validate_sagemaker_sdk_contract


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_text())
    result = validate_sagemaker_sdk_contract(config["training_runtime"]["sdk_version"])
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SAGEMAKER_SDK_V3_CONTRACT_VERIFIED")


if __name__ == "__main__":
    main()
