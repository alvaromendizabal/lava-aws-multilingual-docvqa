from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lava.readers import reader_factory
from lava.readers.model_registry import load_candidates
from lava.readers.qwen35 import (
    _cuda_indices_from_device_map,
    _select_device_map,
)
from lava.readers.schemas import (
    DecodingMode,
    DevicePlacement,
    GenerationSpec,
    ReaderFamily,
    ReaderInputMode,
    ResolvedModel,
)


def _resolved_model(
    *,
    reader_family: ReaderFamily = ReaderFamily.QWEN3_5,
    device_placement: DevicePlacement = DevicePlacement.SINGLE,
    min_cuda_devices: int = 1,
) -> ResolvedModel:
    return ResolvedModel(
        model_key="model",
        model_id="Qwen/Qwen3.5-4B",
        expected_license="apache-2.0",
        expected_pipeline_tag="image-text-to-text",
        parameters_billion=4.0,
        reader_family=reader_family,
        device_placement=device_placement,
        min_cuda_devices=min_cuda_devices,
        trust_remote_code=False,
        instance_type="ml.g7e.48xlarge",
        input_mode=ReaderInputMode.FUSED,
        dtype="bfloat16",
        attention_implementation="sdpa",
        use_kernels=False,
        processor_min_pixels=200704,
        processor_max_pixels=1605632,
        generation=GenerationSpec(
            mode=DecodingMode.DIRECT,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.0,
            seed=1,
        ),
        revision="a" * 40,
        observed_license="apache-2.0",
        observed_pipeline_tag="image-text-to-text",
        resolved_at_utc="2026-09-03T00:00:00+00:00",
        gated=False,
        private=False,
    )


def test_legacy_resolved_model_defaults_to_single_qwen35() -> None:
    payload = _resolved_model().model_dump(mode="json")

    for field in (
        "reader_family",
        "device_placement",
        "min_cuda_devices",
        "trust_remote_code",
    ):
        payload.pop(field)

    model = ResolvedModel.model_validate(payload)

    assert model.reader_family is ReaderFamily.QWEN3_5
    assert model.device_placement is DevicePlacement.SINGLE
    assert model.min_cuda_devices == 1
    assert model.trust_remote_code is False


def test_single_device_policy_allows_multi_gpu_node() -> None:
    model = _resolved_model()

    assert _select_device_map(
        model,
        visible_cuda_devices=8,
    ) == {"": 0}


def test_auto_sharded_policy_requires_and_uses_multiple_gpus() -> None:
    model = _resolved_model(
        device_placement=DevicePlacement.AUTO_SHARDED,
        min_cuda_devices=4,
    )

    assert (
        _select_device_map(
            model,
            visible_cuda_devices=8,
        )
        == "auto"
    )

    with pytest.raises(
        RuntimeError,
        match="Insufficient visible CUDA devices",
    ):
        _select_device_map(
            model,
            visible_cuda_devices=2,
        )


def test_auto_sharded_schema_rejects_single_gpu_contract() -> None:
    with pytest.raises(
        ValueError,
        match="at least two CUDA devices",
    ):
        _resolved_model(
            device_placement=DevicePlacement.AUTO_SHARDED,
            min_cuda_devices=1,
        )


def test_device_map_cuda_index_parser_rejects_offload_names() -> None:
    assert _cuda_indices_from_device_map(
        {
            "vision": 0,
            "layer.0": "cuda:1",
            "layer.1": "cpu",
            "layer.2": "disk",
        }
    ) == (0, 1)


def test_candidate_loader_honors_new_execution_contract(
    tmp_path: Path,
) -> None:
    config = {
        "models": {
            "large": {
                "model_id": "Qwen/Qwen3.5-122B-A10B",
                "expected_license": "apache-2.0",
                "expected_pipeline_tag": "image-text-to-text",
                "parameters_billion": 122.0,
                "reader_family": "qwen3_5",
                "device_placement": "auto_sharded",
                "min_cuda_devices": 4,
                "trust_remote_code": False,
                "instance_type": "ml.g7e.48xlarge",
                "input_mode": "fused",
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
                "use_kernels": False,
                "processor_min_pixels": 200704,
                "processor_max_pixels": 1605632,
                "generation": {
                    "mode": "direct",
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "seed": 1,
                },
            }
        }
    }

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    candidate = load_candidates(path)[0]

    assert candidate.reader_family is ReaderFamily.QWEN3_5
    assert candidate.device_placement is DevicePlacement.AUTO_SHARDED
    assert candidate.min_cuda_devices == 4


def test_reader_factory_is_fail_closed_for_unimplemented_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[str] = []

    class FakeQwenReader:
        def __init__(
            self,
            model_spec: ResolvedModel,
            *,
            region: str,
        ) -> None:
            constructed.append(f"{model_spec.model_key}:{region}")

    monkeypatch.setattr(
        reader_factory,
        "Qwen35Reader",
        FakeQwenReader,
    )

    supported = _resolved_model()

    reader_factory.build_reader(
        supported,
        region="us-west-2",
    )

    assert constructed == ["model:us-west-2"]

    unsupported = _resolved_model(
        reader_family=ReaderFamily.QWEN3_VL,
    )

    with pytest.raises(
        ValueError,
        match="not implemented",
    ):
        reader_factory.build_reader(
            unsupported,
            region="us-west-2",
        )
