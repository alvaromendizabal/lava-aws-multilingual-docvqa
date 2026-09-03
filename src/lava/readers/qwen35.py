"""Qwen3.5 multimodal reader with bounded decoding and detailed telemetry."""

from __future__ import annotations

import hashlib
import io
import os
import random
import time
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np
from PIL import Image

from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.private_artifacts import persist_raw_response
from lava.readers.prompts import SYSTEM_INSTRUCTION, build_reader_instruction
from lava.readers.schemas import (
    OracleExample,
    ReaderInputMode,
    ReaderPrediction,
    ReaderTelemetry,
    ResolvedModel,
)
from lava.readers.structured_output import append_strict_json_instruction


def stable_question_seed(base_seed: int, question_id: str) -> int:
    """Derive a reproducible, question-specific seed without Python hash randomization."""
    digest = hashlib.sha256(f"{base_seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


class Qwen35Reader:
    """Lazy-loaded Qwen3.5 reader for one-GPU SageMaker jobs."""

    def __init__(self, model_spec: ResolvedModel, *, region: str) -> None:
        self.model_spec = model_spec
        self.s3 = boto3.client("s3", region_name=region)
        self._torch: Any | None = None
        self._transformers: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_load_seconds = 0.0

    @staticmethod
    def _split_s3_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    def _get_bytes(self, uri: str) -> bytes:
        bucket, key = self._split_s3_uri(uri)
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()

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

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 oracle benchmark requires a CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Competition-comparable jobs must expose exactly one CUDA GPU")
        self._set_reproducibility(torch, self.model_spec.generation.seed)
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_spec.model_id,
            revision=self.model_spec.revision,
            min_pixels=self.model_spec.processor_min_pixels,
            max_pixels=self.model_spec.processor_max_pixels,
        )
        model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if model_class is None:
            model_class = transformers.AutoModelForMultimodalLM
        dtype = getattr(torch, self.model_spec.dtype)
        model_kwargs: dict[str, Any] = {
            "revision": self.model_spec.revision,
            "dtype": dtype,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "attn_implementation": self.model_spec.attention_implementation,
        }
        if self.model_spec.use_kernels:
            model_kwargs["use_kernels"] = True
        model = model_class.from_pretrained(self.model_spec.model_id, **model_kwargs)
        model.eval()
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Model was not loaded completely on the single CUDA GPU")
        device_map = getattr(model, "hf_device_map", {})
        if any(str(device).casefold() in {"cpu", "disk", "meta"} for device in device_map.values()):
            raise RuntimeError("CPU or disk model offload is prohibited in the benchmark track")
        self._torch = torch
        self._transformers = transformers
        self._processor = processor
        self._model = model
        self._model_load_seconds = time.perf_counter() - started

    def _messages(self, example: OracleExample) -> tuple[list[dict[str, Any]], int, int]:
        if self._processor is None:
            raise RuntimeError("Reader is not loaded")
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
                with Image.open(io.BytesIO(self._get_bytes(page.image_s3_uri))) as source:
                    image = source.convert("RGB")
                image_count += 1
                total_pixels += image.width * image.height
                content.append({"type": "image", "image": image})
            if include_text:
                native_text = self._get_bytes(page.text_s3_uri).decode(
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
                    available_pages=example.evidence_pages,
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

    def predict(self, example: OracleExample) -> tuple[ReaderPrediction, ReaderTelemetry]:
        """Generate and parse one structured answer."""
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
        torch.cuda.reset_peak_memory_stats()
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
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        prompt_length = int(inputs["input_ids"].shape[-1])
        output_ids = generated[:, prompt_length:]
        raw_response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        persist_raw_response(raw_response)
        try:
            prediction = parse_reader_response(
                question_id=example.question_id,
                answer_format=example.answer_format,
                raw_response=raw_response,
                allowed_pages=example.evidence_pages,
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
        major, minor = torch.cuda.get_device_capability(0)
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
            peak_cuda_memory_allocated_mib=torch.cuda.max_memory_allocated() / (1024**2),
            peak_cuda_memory_reserved_mib=torch.cuda.max_memory_reserved() / (1024**2),
            gpu_name=torch.cuda.get_device_name(0),
            cuda_compute_capability=f"{major}.{minor}",
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            dtype=self.model_spec.dtype,
            attention_implementation=self.model_spec.attention_implementation,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            template_switch_supported=True,
        )
        return prediction, telemetry
