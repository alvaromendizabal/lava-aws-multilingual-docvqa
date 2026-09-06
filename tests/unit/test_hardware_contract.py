from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "qwen35_4b_fused_direct": "ml.g5.2xlarge",
    "qwen35_9b_fused_direct": "ml.g6e.2xlarge",
    "qwen38_27b_fused_direct": "ml.g7e.12xlarge",
}


def test_config_pins_validated_lava_hardware() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/oracle_reader_benchmark.yaml").read_text(encoding="utf-8")
    )
    models = config["models"]
    for model_key, expected_instance in EXPECTED.items():
        assert models[model_key]["instance_type"] == expected_instance


def test_model_lock_matches_hardware_contract() -> None:
    lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text(encoding="utf-8"))
    resolved = {item["model_key"]: item for item in lock["resolved_models"]}
    for model_key, expected_instance in EXPECTED.items():
        assert resolved[model_key]["instance_type"] == expected_instance


def test_27b_remains_single_gpu_high_memory_contract() -> None:
    lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text(encoding="utf-8"))
    resolved = {item["model_key"]: item for item in lock["resolved_models"]}
    model = resolved["qwen38_27b_fused_direct"]
    assert model["device_placement"] == "single"
    assert model["min_cuda_devices"] == 1
    assert model["min_cuda_memory_per_device_gib"] >= 80
    assert model["instance_type"] == "ml.g7e.12xlarge"
