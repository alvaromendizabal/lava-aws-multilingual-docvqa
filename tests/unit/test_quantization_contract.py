from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from lava.readers.qwen35 import _build_quantization_config
from lava.readers.schemas import QuantizationMode

ROOT = Path(__file__).resolve().parents[2]
INT8_KEY = "qwen38_27b_int8_fused_direct"
NF4_KEY = "qwen38_27b_nf4_fused_direct"
BF16_KEY = "qwen38_27b_fused_direct"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _fake_transformers() -> SimpleNamespace:
    return SimpleNamespace(BitsAndBytesConfig=_FakeBitsAndBytesConfig)


def test_quantized_27b_candidates_target_g6e_without_mutating_bf16() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/oracle_reader_benchmark.yaml").read_text(encoding="utf-8")
    )
    models = config["models"]

    assert models[BF16_KEY]["instance_type"] == "ml.g7e.12xlarge"
    assert models[BF16_KEY].get("quantization", "none") == "none"

    assert models[INT8_KEY]["instance_type"] == "ml.g6e.2xlarge"
    assert models[INT8_KEY]["quantization"] == "int8"
    assert models[INT8_KEY]["device_placement"] == "single"
    assert models[INT8_KEY]["min_cuda_devices"] == 1
    assert models[INT8_KEY]["min_cuda_memory_per_device_gib"] == 40

    assert models[NF4_KEY]["instance_type"] == "ml.g6e.2xlarge"
    assert models[NF4_KEY]["quantization"] == "nf4"
    assert models[NF4_KEY]["device_placement"] == "single"
    assert models[NF4_KEY]["min_cuda_devices"] == 1
    assert models[NF4_KEY]["min_cuda_memory_per_device_gib"] == 32


def test_registry_reuses_frozen_27b_revision() -> None:
    lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text(encoding="utf-8"))
    resolved = {row["model_key"]: row for row in lock["resolved_models"]}

    for key, mode in ((INT8_KEY, "int8"), (NF4_KEY, "nf4")):
        assert resolved[key]["model_id"] == "Qwen/Qwen3.8-27B"
        assert resolved[key]["revision"] == REVISION
        assert resolved[key]["instance_type"] == "ml.g6e.2xlarge"
        assert resolved[key]["quantization"] == mode

    assert resolved[BF16_KEY]["revision"] == REVISION
    assert resolved[BF16_KEY]["instance_type"] == "ml.g7e.12xlarge"


def test_int8_config_is_explicit_and_disables_cpu_offload() -> None:
    spec = SimpleNamespace(quantization=QuantizationMode.INT8, dtype="bfloat16")
    torch = SimpleNamespace(bfloat16=object())
    config = _build_quantization_config(
        spec,  # type: ignore[arg-type]
        torch=torch,
        transformers=_fake_transformers(),
    )
    assert config is not None
    assert config.kwargs == {
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "llm_int8_enable_fp32_cpu_offload": False,
    }


def test_nf4_config_uses_bfloat16_compute_and_double_quant() -> None:
    compute_dtype = object()
    spec = SimpleNamespace(quantization=QuantizationMode.NF4, dtype="bfloat16")
    torch = SimpleNamespace(bfloat16=compute_dtype)
    config = _build_quantization_config(
        spec,  # type: ignore[arg-type]
        torch=torch,
        transformers=_fake_transformers(),
    )
    assert config is not None
    assert config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }


def test_none_mode_has_no_quantization_config() -> None:
    spec = SimpleNamespace(quantization=QuantizationMode.NONE, dtype="bfloat16")
    torch = SimpleNamespace(bfloat16=object())
    config = _build_quantization_config(
        spec,  # type: ignore[arg-type]
        torch=torch,
        transformers=_fake_transformers(),
    )
    assert config is None


def test_gpu_requirements_pin_bitsandbytes() -> None:
    requirements = (ROOT / "pipelines/oracle_reader/requirements-gpu.txt").read_text(
        encoding="utf-8"
    )
    assert "bitsandbytes==0.50.2\n" in requirements


def test_public_summary_records_quantization_provenance() -> None:
    source = (ROOT / "src/lava/readers/benchmark.py").read_text(encoding="utf-8")
    assert '"quantization": model_spec.quantization.value' in source
    assert '"dtype": model_spec.dtype' in source


def test_sagemaker_plan_exposes_quantization_before_paid_submission() -> None:
    source = (ROOT / "src/lava/readers/sagemaker.py").read_text(encoding="utf-8")
    assert "quantization=model.quantization" in source
    assert "dtype=model.dtype" in source


def test_nf4_g5_candidate_uses_distinct_capacity_pool() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/oracle_reader_benchmark.yaml").read_text(encoding="utf-8")
    )
    model = config["models"]["qwen38_27b_nf4_g5_fused_direct"]
    assert model["model_id"] == "Qwen/Qwen3.8-27B"
    assert model["instance_type"] == "ml.g5.2xlarge"
    assert model["quantization"] == "nf4"
    assert model["device_placement"] == "single"
    assert model["min_cuda_devices"] == 1
    assert model["min_cuda_memory_per_device_gib"] == 20

    lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text(encoding="utf-8"))
    resolved = {row["model_key"]: row for row in lock["resolved_models"]}
    row = resolved["qwen38_27b_nf4_g5_fused_direct"]
    assert row["revision"] == REVISION
    assert row["instance_type"] == "ml.g5.2xlarge"
    assert row["quantization"] == "nf4"
