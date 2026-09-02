"""Strict contracts for oracle-evidence reader experiments."""

from __future__ import annotations

import math
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lava.evaluation.schemas import AnswerFormat

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FrozenModel(BaseModel):
    """Immutable strict Pydantic base model."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ReaderInputMode(StrEnum):
    """Evidence representation supplied to the reader."""

    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    FUSED = "fused"


class DecodingMode(StrEnum):
    """Qwen response mode."""

    DIRECT = "direct"
    THINKING = "thinking"


class GenerationSpec(FrozenModel):
    """Pinned generation configuration."""

    mode: DecodingMode
    max_new_tokens: int = Field(ge=1, le=4096)
    do_sample: bool = False
    temperature: float | None = Field(default=None, gt=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.8, le=2.0)
    seed: int = Field(ge=0)

    @property
    def thinking(self) -> bool:
        """Return whether the model should use its thinking template."""
        return self.mode is DecodingMode.THINKING

    @model_validator(mode="after")
    def validate_sampling(self) -> GenerationSpec:
        """Enforce an explicit and reproducible decoding policy."""
        sampling_values = (self.temperature, self.top_p, self.top_k, self.min_p)
        if self.do_sample and any(value is None for value in sampling_values):
            raise ValueError("Sampled decoding requires temperature, top_p, top_k, and min_p")
        if not self.do_sample and any(value is not None for value in sampling_values):
            raise ValueError("Deterministic decoding cannot include sampling parameters")
        if self.thinking and not self.do_sample:
            raise ValueError("Qwen thinking mode must use bounded sampled decoding")
        return self


class ModelCandidate(FrozenModel):
    """Unresolved model candidate loaded from benchmark configuration."""

    model_key: str = Field(min_length=1)
    model_id: str = Field(pattern=r"^[^/]+/[^/]+$")
    expected_license: str = Field(min_length=1)
    expected_pipeline_tag: str = Field(min_length=1)
    parameters_billion: float = Field(gt=0.0)
    instance_type: str = Field(pattern=r"^ml\.[a-z0-9.]+$")
    input_mode: ReaderInputMode
    dtype: str = Field(pattern=r"^(bfloat16|float16)$")
    attention_implementation: str = Field(pattern=r"^(sdpa|eager|flash_attention_2)$")
    use_kernels: bool = False
    processor_min_pixels: int = Field(ge=28 * 28)
    processor_max_pixels: int = Field(ge=28 * 28)
    generation: GenerationSpec

    @model_validator(mode="after")
    def validate_pixels(self) -> ModelCandidate:
        """Require a valid visual-token budget."""
        if self.processor_max_pixels < self.processor_min_pixels:
            raise ValueError("processor_max_pixels must be at least processor_min_pixels")
        return self


class ResolvedModel(ModelCandidate):
    """Candidate locked to an immutable Hugging Face revision."""

    revision: str
    observed_license: str
    observed_pipeline_tag: str
    resolved_at_utc: str
    last_modified: str | None = None
    gated: bool = False
    private: bool = False

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        """Require a full immutable Git commit SHA."""
        if not _SHA40.fullmatch(value):
            raise ValueError("Model revision must be a 40-character lowercase Git SHA")
        return value

    @model_validator(mode="after")
    def validate_access(self) -> ResolvedModel:
        """Reject gated or private models from the competition-comparable track."""
        if self.gated or self.private:
            raise ValueError("Competition-comparable candidates must be public and ungated")
        return self


class OraclePageAsset(FrozenModel):
    """One rendered and structurally extracted gold evidence page."""

    asset_version: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_alias: str = Field(pattern=r"^doc-[0-9]{2}$")
    page_number: int = Field(ge=1)
    source_pdf_s3_uri: str = Field(pattern=r"^s3://")
    source_pdf_sha256: str
    source_pdf_version_id: str | None = None
    source_pdf_etag: str | None = None
    image_s3_uri: str = Field(pattern=r"^s3://")
    image_sha256: str
    image_version_id: str | None = None
    text_s3_uri: str = Field(pattern=r"^s3://")
    text_sha256: str
    text_version_id: str | None = None
    layout_s3_uri: str = Field(pattern=r"^s3://")
    layout_sha256: str
    layout_version_id: str | None = None
    width_pixels: int = Field(ge=1)
    height_pixels: int = Field(ge=1)
    dpi: int = Field(ge=72, le=600)
    native_text_characters: int = Field(ge=0)
    word_count: int = Field(ge=0)
    text_block_count: int = Field(ge=0)
    embedded_image_count: int = Field(ge=0)

    @field_validator(
        "source_pdf_sha256",
        "image_sha256",
        "text_sha256",
        "layout_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require lowercase SHA-256 values."""
        if not _SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase 64-character SHA-256")
        return value


class OracleExample(FrozenModel):
    """Private oracle-evidence example containing one labeled question."""

    protocol_lock_id: str
    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_alias: str = Field(pattern=r"^doc-[0-9]{2}$")
    question: str = Field(min_length=1)
    answer_format: AnswerFormat
    answer: str = Field(min_length=1)
    evidence_pages: tuple[int, ...] = Field(min_length=1, max_length=4)
    language: str = Field(min_length=2, max_length=8)
    pages: tuple[OraclePageAsset, ...] = Field(min_length=1, max_length=4)

    @field_validator("protocol_lock_id")
    @classmethod
    def validate_lock(cls, value: str) -> str:
        """Require the immutable evaluation lock."""
        if not _SHA256.fullmatch(value):
            raise ValueError("protocol_lock_id must be a SHA-256")
        return value

    @model_validator(mode="after")
    def validate_page_alignment(self) -> OracleExample:
        """Ensure assets exactly match the gold evidence-page set."""
        asset_pages = tuple(sorted(asset.page_number for asset in self.pages))
        if asset_pages != self.evidence_pages:
            raise ValueError("Page assets must exactly match the gold evidence pages")
        if any(asset.document_id != self.document_id for asset in self.pages):
            raise ValueError("All page assets must belong to the example document")
        return self


class ReaderPrediction(FrozenModel):
    """Parsed structured reader output without hidden reasoning text."""

    question_id: str = Field(min_length=1)
    answer_format: AnswerFormat
    answer: str
    evidence_pages: tuple[int, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    schema_valid: bool
    parser_error_code: str | None = None
    raw_response_sha256: str

    @field_validator("evidence_pages")
    @classmethod
    def validate_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require sorted unique positive evidence pages."""
        if any(page < 1 for page in value):
            raise ValueError("Evidence pages must be positive")
        if value != tuple(sorted(set(value))):
            raise ValueError("Evidence pages must be unique and sorted")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Reject non-finite confidence values."""
        if not math.isfinite(value):
            raise ValueError("Confidence must be finite")
        return value

    @field_validator("raw_response_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256."""
        if not _SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def validate_abstention(self) -> ReaderPrediction:
        """Require abstentions to contain no answer or evidence claim."""
        if self.abstain and (self.answer not in {"", "[]"} or self.evidence_pages):
            raise ValueError("Abstentions must contain an empty answer and no evidence pages")
        return self


class ReaderTelemetry(FrozenModel):
    """Per-question runtime, visual-input, and hardware measurements."""

    model_load_seconds: float = Field(ge=0.0)
    preprocessing_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    total_seconds: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    image_count: int = Field(ge=0)
    total_image_pixels: int = Field(ge=0)
    raw_response_characters: int = Field(ge=0)
    peak_cuda_memory_allocated_mib: float = Field(ge=0.0)
    peak_cuda_memory_reserved_mib: float = Field(ge=0.0)
    gpu_name: str
    cuda_compute_capability: str
    torch_version: str
    transformers_version: str
    dtype: str
    attention_implementation: str
    deterministic_algorithms_enabled: bool
    template_switch_supported: bool


class BenchmarkRecord(FrozenModel):
    """Private per-question oracle benchmark record."""

    experiment_id: str = Field(min_length=1)
    protocol_lock_id: str
    asset_manifest_sha256: str
    git_commit_sha: str
    prompt_version: str
    model_key: str
    model_id: str
    model_revision: str
    question_id: str
    document_alias: str
    language: str
    answer_format: AnswerFormat
    gold_evidence_pages: tuple[int, ...]
    prediction: ReaderPrediction
    telemetry: ReaderTelemetry
    normalized_exact_answer_score: float = Field(ge=0.0, le=1.0)
    self_grounding_f1: float = Field(ge=0.0, le=1.0)
    oracle_fixed_overall_diagnostic: float = Field(ge=0.0, le=1.0)


class SageMakerJobPlan(FrozenModel):
    """Charge-bounded SageMaker job plan shown before submission."""

    sdk_version: str
    model_key: str
    model_id: str
    model_revision: str
    protocol_lock_id: str
    git_commit_sha: str
    bucket: str
    manifest_s3_uri: str = Field(pattern=r"^s3://")
    output_s3_prefix: str = Field(pattern=r"^s3://")
    training_image: str
    training_image_digest: str
    instance_type: str
    instance_count: int = Field(ge=1, le=1)
    volume_size_gb: int = Field(ge=30, le=500)
    max_runtime_seconds: int = Field(ge=60, le=3600)
    max_wait_seconds: int = Field(ge=60, le=7200)
    managed_spot: bool
    limit: int = Field(ge=1, le=16)
    input_mode: ReaderInputMode
    generation: GenerationSpec
    creates_endpoint: bool = False

    @field_validator("protocol_lock_id", "training_image_digest")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Validate lock and digest strings."""
        candidate = value.removeprefix("sha256:")
        if not _SHA256.fullmatch(candidate):
            raise ValueError("Expected SHA-256 digest")
        return value
