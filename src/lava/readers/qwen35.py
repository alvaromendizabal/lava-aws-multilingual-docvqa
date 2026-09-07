"""Qwen3.5 multimodal reader with bounded decoding and detailed telemetry."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import random
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np
from PIL import Image

from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.private_artifacts import persist_raw_response
from lava.readers.prompts import SYSTEM_INSTRUCTION, build_reader_instruction
from lava.readers.schemas import (
    DevicePlacement,
    OracleExample,
    QuantizationMode,
    ReaderInput,
    ReaderInputMode,
    ReaderPrediction,
    ReaderTelemetry,
    ResolvedModel,
)
from lava.readers.structured_output import append_strict_json_instruction

_BITSANDBYTES_VERSION = "0.50.2"


def _validate_quantization_runtime(mode: QuantizationMode) -> str | None:
    """Import and version-check the pinned quantization backend before model download."""
    if mode is QuantizationMode.NONE:
        return None

    try:
        importlib.import_module("bitsandbytes")
        observed = version("bitsandbytes")
    except (ImportError, OSError, PackageNotFoundError) as error:
        raise RuntimeError(
            "Quantized reader requires a working bitsandbytes installation"
        ) from error

    if observed != _BITSANDBYTES_VERSION:
        raise RuntimeError(
            "Unexpected bitsandbytes version: "
            f"expected={_BITSANDBYTES_VERSION}, observed={observed}"
        )
    return observed


def _build_quantization_config(
    model_spec: ResolvedModel,
    *,
    torch: Any,
    transformers: Any,
) -> Any | None:
    """Build one explicit Transformers quantization configuration."""
    mode = model_spec.quantization
    if mode is QuantizationMode.NONE:
        return None

    config_class = getattr(transformers, "BitsAndBytesConfig", None)
    if config_class is None:
        raise RuntimeError("Installed Transformers does not expose BitsAndBytesConfig")

    if mode is QuantizationMode.INT8:
        return config_class(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    if mode is QuantizationMode.NF4:
        compute_dtype = getattr(torch, model_spec.dtype)
        return config_class(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    raise RuntimeError(f"Unsupported quantization mode: {mode!r}")


def _cuda_indices_from_device_map(
    device_map: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return sorted CUDA indices referenced by a Transformers device map."""
    indices: set[int] = set()

    for device in device_map.values():
        if isinstance(device, bool):
            continue

        if isinstance(device, int):
            if device >= 0:
                indices.add(device)
            continue

        value = str(device).strip().casefold()

        if value == "cuda":
            indices.add(0)
            continue

        if value.startswith("cuda:"):
            suffix = value.removeprefix("cuda:")

            if suffix.isdigit():
                indices.add(int(suffix))

    return tuple(sorted(indices))


def _select_device_map(
    model_spec: ResolvedModel,
    *,
    visible_cuda_devices: int,
) -> dict[str, int] | str:
    """Select an explicit placement strategy and fail before model loading."""
    if visible_cuda_devices < model_spec.min_cuda_devices:
        raise RuntimeError(
            "Insufficient visible CUDA devices for model contract: "
            f"required={model_spec.min_cuda_devices}, "
            f"visible={visible_cuda_devices}"
        )

    if model_spec.device_placement is DevicePlacement.SINGLE:
        return {"": 0}

    if model_spec.device_placement is DevicePlacement.AUTO_SHARDED:
        return "auto"

    raise RuntimeError(f"Unsupported device placement: {model_spec.device_placement!r}")


def stable_question_seed(base_seed: int, question_id: str) -> int:
    """Derive a reproducible, question-specific seed without Python hash randomization."""
    digest = hashlib.sha256(f"{base_seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


class Qwen35Reader:
    """Lazy-loaded Qwen3.5-family reader for one SageMaker GPU node."""

    def __init__(self, model_spec: ResolvedModel, *, region: str) -> None:
        self.model_spec = model_spec
        self.s3 = boto3.client("s3", region_name=region)
        self._torch: Any | None = None
        self._transformers: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_load_seconds = 0.0
        self._active_cuda_indices: tuple[int, ...] = ()
        self._bitsandbytes_version: str | None = None

    @staticmethod
    def _split_s3_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    def _get_bytes(self, uri: str, *, expected_sha256: str, version_id: str | None = None) -> bytes:
        """Verify exactly the frozen input bytes before constructing a model prompt."""
        bucket, key = self._split_s3_uri(uri)
        request: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if version_id is not None:
            request["VersionId"] = version_id
        response = self.s3.get_object(**request)
        with response["Body"] as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Reader asset checksum mismatch")
        if version_id is not None and response.get("VersionId") != version_id:
            raise ValueError("Reader asset version mismatch")
        return payload

    @staticmethod
    def _set_reproducibility(torch: Any, seed: int) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("high")

    def load(self) -> None:
        """Load one pinned checkpoint wholly onto one CUDA device."""
        if self._model is not None:
            return
        started = time.perf_counter()
        import torch
        import transformers

        bitsandbytes_version = _validate_quantization_runtime(self.model_spec.quantization)

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen oracle benchmark requires CUDA")

        visible_cuda_devices = int(torch.cuda.device_count())

        device_map = _select_device_map(
            self.model_spec,
            visible_cuda_devices=visible_cuda_devices,
        )

        self._set_reproducibility(
            torch,
            self.model_spec.generation.seed,
        )

        processor_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "min_pixels": self.model_spec.processor_min_pixels,
            "max_pixels": self.model_spec.processor_max_pixels,
        }

        if self.model_spec.trust_remote_code:
            processor_kwargs["trust_remote_code"] = True

        processor = transformers.AutoProcessor.from_pretrained(
            self.model_spec.model_id,
            **processor_kwargs,
        )
        model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if model_class is None:
            model_class = transformers.AutoModelForMultimodalLM
        dtype = getattr(torch, self.model_spec.dtype)
        quantization_config = _build_quantization_config(
            self.model_spec,
            torch=torch,
            transformers=transformers,
        )
        model_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "attn_implementation": self.model_spec.attention_implementation,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.model_spec.use_kernels:
            model_kwargs["use_kernels"] = True

        if self.model_spec.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        model = model_class.from_pretrained(
            self.model_spec.model_id,
            **model_kwargs,
        )
        model.eval()

        primary_device = next(model.parameters()).device

        if primary_device.type != "cuda":
            raise RuntimeError("Model primary parameters were not loaded on CUDA")

        hf_device_map = getattr(model, "hf_device_map", {})

        if not isinstance(hf_device_map, Mapping):
            raise TypeError("Transformers returned a non-mapping hf_device_map")

        offloaded_devices = {
            str(device).strip().casefold()
            for device in hf_device_map.values()
            if str(device).strip().casefold() in {"cpu", "disk", "meta"}
        }

        if offloaded_devices:
            raise RuntimeError(
                "CPU, disk, or meta-device model offload is prohibited: "
                f"{sorted(offloaded_devices)}"
            )

        active_cuda_indices = _cuda_indices_from_device_map(hf_device_map)

        if self.model_spec.device_placement is DevicePlacement.SINGLE:
            primary_index = 0 if primary_device.index is None else int(primary_device.index)

            if primary_index != 0:
                raise RuntimeError("Single-device placement must load on cuda:0")

            active_cuda_indices = (0,)

        else:
            if not hf_device_map:
                raise RuntimeError(
                    "Auto-sharded placement requires an explicit Transformers hf_device_map"
                )

            if len(active_cuda_indices) < self.model_spec.min_cuda_devices:
                raise RuntimeError(
                    "Auto-sharded model did not use the minimum CUDA "
                    "device count: "
                    f"required={self.model_spec.min_cuda_devices}, "
                    f"used={active_cuda_indices}"
                )

        self._active_cuda_indices = active_cuda_indices
        self._torch = torch
        self._transformers = transformers
        self._processor = processor
        self._model = model
        self._bitsandbytes_version = bitsandbytes_version
        self._model_load_seconds = time.perf_counter() - started

    def _messages(
        self, example: OracleExample | ReaderInput
    ) -> tuple[list[dict[str, Any]], int, int]:
        if self._processor is None:
            raise RuntimeError("Reader is not loaded")
        if isinstance(example, OracleExample):
            example = ReaderInput.from_oracle(example)
        content: list[dict[str, Any]] = []
        include_images = self.model_spec.input_mode in {
            ReaderInputMode.IMAGE_ONLY,
            ReaderInputMode.FUSED,
        }
        include_text = self.model_spec.input_mode in {
            ReaderInputMode.TEXT_ONLY,
            ReaderInputMode.FUSED,
        }
        image_count = 0
        total_pixels = 0
        for page in example.pages:
            content.append({"type": "text", "text": f"[PAGE {page.page_number}]"})
            if include_images:
                with Image.open(
                    io.BytesIO(
                        self._get_bytes(
                            page.image_s3_uri,
                            expected_sha256=page.image_sha256,
                            version_id=page.image_version_id,
                        )
                    )
                ) as source:
                    image = source.convert("RGB")
                image_count += 1
                total_pixels += image.width * image.height
                content.append({"type": "image", "image": image})
            if include_text:
                native_text = self._get_bytes(
                    page.text_s3_uri,
                    expected_sha256=page.text_sha256,
                    version_id=page.text_version_id,
                ).decode(
                    "utf-8",
                    errors="replace",
                )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Native PDF text for page {page.page_number}:\n{native_text[:16000]}"
                        ),
                    }
                )
        content.append(
            {
                "type": "text",
                "text": build_reader_instruction(
                    question=example.question,
                    language=example.language,
                    answer_format=example.answer_format,
                    available_pages=example.available_pages,
                ),
            }
        )
        return (
            [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": content},
            ],
            image_count,
            total_pixels,
        )

    def _template_inputs(self, messages: list[dict[str, Any]]) -> Any:
        processor = self._processor
        if processor is None:
            raise RuntimeError("Reader is not loaded")
        thinking = self.model_spec.generation.thinking
        common = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            messages = append_strict_json_instruction(messages)
            return processor.apply_chat_template(messages, **common, enable_thinking=thinking)
        except TypeError:
            try:
                return processor.apply_chat_template(messages, enable_thinking=thinking, **common)
            except TypeError as second_error:
                raise RuntimeError(
                    "Installed Transformers cannot explicitly control Qwen3.5 thinking mode"
                ) from second_error

    def predict(
        self, example: OracleExample | ReaderInput
    ) -> tuple[ReaderPrediction, ReaderTelemetry]:
        """Generate and parse one structured answer."""
        if isinstance(example, OracleExample):
            example = ReaderInput.from_oracle(example)
        total_started = time.perf_counter()
        self.load()
        torch = self._torch
        transformers = self._transformers
        processor = self._processor
        model = self._model
        if torch is None or transformers is None or processor is None or model is None:
            raise RuntimeError("Reader failed to initialize")
        effective_seed = stable_question_seed(
            self.model_spec.generation.seed,
            example.question_id,
        )
        self._set_reproducibility(torch, effective_seed)
        active_cuda_indices = self._active_cuda_indices or (0,)

        for device_index in active_cuda_indices:
            torch.cuda.reset_peak_memory_stats(device_index)
        preprocess_started = time.perf_counter()
        messages, image_count, total_pixels = self._messages(example)
        inputs = self._template_inputs(messages)
        model_device = next(model.parameters()).device
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        preprocessing_seconds = time.perf_counter() - preprocess_started
        generation: dict[str, Any] = {
            "max_new_tokens": self.model_spec.generation.max_new_tokens,
            "do_sample": self.model_spec.generation.do_sample,
            "repetition_penalty": self.model_spec.generation.repetition_penalty,
            "use_cache": True,
        }
        if self.model_spec.generation.do_sample:
            generation.update(
                temperature=self.model_spec.generation.temperature,
                top_p=self.model_spec.generation.top_p,
                top_k=self.model_spec.generation.top_k,
                min_p=self.model_spec.generation.min_p,
            )
        eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        if eos_token_id is not None:
            generation["pad_token_id"] = eos_token_id
        for device_index in active_cuda_indices:
            torch.cuda.synchronize(device_index)

        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation)
        for device_index in active_cuda_indices:
            torch.cuda.synchronize(device_index)

        generation_seconds = time.perf_counter() - generation_started
        prompt_length = int(inputs["input_ids"].shape[-1])
        output_ids = generated[:, prompt_length:]
        raw_response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        persist_raw_response(raw_response, question_id=example.question_id)
        try:
            prediction = parse_reader_response(
                question_id=example.question_id,
                answer_format=example.answer_format,
                raw_response=raw_response,
                allowed_pages=example.available_pages,
            )
        except ReaderOutputError as error:
            prediction = ReaderPrediction(
                question_id=example.question_id,
                answer_format=example.answer_format,
                answer="",
                evidence_pages=(),
                confidence=0.0,
                abstain=True,
                schema_valid=False,
                parser_error_code=error.code,
                raw_response_sha256=hashlib.sha256(raw_response.encode()).hexdigest(),
            )
        peak_allocated_mib = sum(
            torch.cuda.max_memory_allocated(device_index) for device_index in active_cuda_indices
        ) / (1024**2)

        peak_reserved_mib = sum(
            torch.cuda.max_memory_reserved(device_index) for device_index in active_cuda_indices
        ) / (1024**2)

        gpu_names = [
            f"cuda:{device_index}={torch.cuda.get_device_name(device_index)}"
            for device_index in active_cuda_indices
        ]

        capabilities = []

        for device_index in active_cuda_indices:
            major, minor = torch.cuda.get_device_capability(device_index)
            capabilities.append(f"cuda:{device_index}={major}.{minor}")

        telemetry = ReaderTelemetry(
            model_load_seconds=self._model_load_seconds,
            preprocessing_seconds=preprocessing_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - total_started,
            prompt_tokens=prompt_length,
            generated_tokens=int(output_ids.shape[-1]),
            image_count=image_count,
            total_image_pixels=total_pixels,
            raw_response_characters=len(raw_response),
            peak_cuda_memory_allocated_mib=peak_allocated_mib,
            peak_cuda_memory_reserved_mib=peak_reserved_mib,
            gpu_name=" | ".join(gpu_names),
            cuda_compute_capability=" | ".join(capabilities),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            dtype=self.model_spec.dtype,
            quantization=self.model_spec.quantization,
            bitsandbytes_version=self._bitsandbytes_version,
            attention_implementation=self.model_spec.attention_implementation,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            template_switch_supported=True,
        )
        return prediction, telemetry
