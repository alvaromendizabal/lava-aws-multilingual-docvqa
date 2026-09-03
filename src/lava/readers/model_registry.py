"""Resolve mutable model names to immutable Hugging Face revisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

from lava.readers.schemas import (
    DecodingMode,
    DevicePlacement,
    GenerationSpec,
    ModelCandidate,
    ReaderFamily,
    ReaderInputMode,
    ResolvedModel,
)

JsonFetcher = Callable[[str], Mapping[str, Any]]


def load_candidates(path: Path) -> tuple[ModelCandidate, ...]:
    """Load and validate candidate definitions from benchmark YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise TypeError("Benchmark configuration must contain a models mapping")
    candidates: list[ModelCandidate] = []
    for model_key, values in payload["models"].items():
        generation_values = values["generation"]
        generation = GenerationSpec(
            mode=DecodingMode(generation_values["mode"]),
            max_new_tokens=int(generation_values["max_new_tokens"]),
            do_sample=bool(generation_values["do_sample"]),
            temperature=generation_values.get("temperature"),
            top_p=generation_values.get("top_p"),
            top_k=generation_values.get("top_k"),
            min_p=generation_values.get("min_p"),
            repetition_penalty=float(generation_values.get("repetition_penalty", 1.0)),
            seed=int(generation_values["seed"]),
        )
        candidates.append(
            ModelCandidate(
                model_key=str(model_key),
                model_id=values["model_id"],
                expected_license=values["expected_license"],
                expected_pipeline_tag=values["expected_pipeline_tag"],
                parameters_billion=float(values["parameters_billion"]),
                reader_family=ReaderFamily(values.get("reader_family", ReaderFamily.QWEN3_5.value)),
                device_placement=DevicePlacement(
                    values.get("device_placement", DevicePlacement.SINGLE.value)
                ),
                min_cuda_devices=int(values.get("min_cuda_devices", 1)),
                min_cuda_memory_per_device_gib=int(values.get("min_cuda_memory_per_device_gib", 1)),
                min_total_cuda_memory_gib=int(values.get("min_total_cuda_memory_gib", 1)),
                trust_remote_code=bool(values.get("trust_remote_code", False)),
                instance_type=values["instance_type"],
                input_mode=ReaderInputMode(values["input_mode"]),
                dtype=values["dtype"],
                attention_implementation=values["attention_implementation"],
                use_kernels=bool(values.get("use_kernels", False)),
                processor_min_pixels=int(values["processor_min_pixels"]),
                processor_max_pixels=int(values["processor_max_pixels"]),
                generation=generation,
            )
        )
    if len(candidates) != len({item.model_key for item in candidates}):
        raise ValueError("Model keys must be unique")
    return tuple(candidates)


def _default_fetcher(model_id: str) -> Mapping[str, Any]:
    with requests.Session() as session:
        response = session.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=30,
            headers={"Accept": "application/json"},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Hugging Face model API returned a non-object payload")
    return payload


def resolve_candidate(
    candidate: ModelCandidate,
    *,
    fetcher: JsonFetcher = _default_fetcher,
    resolved_at_utc: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ResolvedModel:
    """Validate repository metadata and lock one candidate to a commit SHA."""
    payload = metadata or fetcher(candidate.model_id)
    revision = str(payload.get("sha", "")).lower()
    card_data = payload.get("cardData") or {}
    if not isinstance(card_data, Mapping):
        card_data = {}
    observed_license = str(card_data.get("license", payload.get("license", ""))).casefold()
    observed_pipeline_tag = str(payload.get("pipeline_tag", ""))
    if observed_license != candidate.expected_license.casefold():
        raise ValueError(f"Unexpected license for {candidate.model_id}: {observed_license!r}")
    if observed_pipeline_tag != candidate.expected_pipeline_tag:
        raise ValueError(
            f"Unexpected pipeline tag for {candidate.model_id}: {observed_pipeline_tag!r}"
        )
    return ResolvedModel(
        **candidate.model_dump(),
        revision=revision,
        observed_license=observed_license,
        observed_pipeline_tag=observed_pipeline_tag,
        resolved_at_utc=resolved_at_utc or datetime.now(UTC).isoformat(),
        last_modified=(str(payload["lastModified"]) if payload.get("lastModified") else None),
        gated=bool(payload.get("gated", False)),
        private=bool(payload.get("private", False)),
    )


def _canonical_registry_body(body: Mapping[str, Any]) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def resolve_registry(config_path: Path, output_path: Path) -> dict[str, Any]:
    """Resolve candidates and write a canonical immutable revision lock."""
    candidates = load_candidates(config_path)
    cache: dict[str, Mapping[str, Any]] = {}
    timestamp = datetime.now(UTC).isoformat()
    resolved: list[ResolvedModel] = []
    for candidate in candidates:
        if candidate.model_id not in cache:
            cache[candidate.model_id] = _default_fetcher(candidate.model_id)
        resolved.append(
            resolve_candidate(
                candidate,
                resolved_at_utc=timestamp,
                metadata=cache[candidate.model_id],
            )
        )
    body = {
        "schema_version": 2,
        "generated_at_utc": timestamp,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidate_count": len(resolved),
        "unique_model_repository_count": len(cache),
        "resolved_models": [model.model_dump(mode="json") for model in resolved],
    }
    lock = {
        **body,
        "registry_sha256": hashlib.sha256(_canonical_registry_body(body)).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return lock


def load_resolved_model(path: Path, model_key: str) -> ResolvedModel:
    """Load one model from a registry lock after verifying the lock hash."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("registry_sha256")
    body = {key: value for key, value in payload.items() if key != "registry_sha256"}
    observed = hashlib.sha256(_canonical_registry_body(body)).hexdigest()
    if expected != observed:
        raise ValueError("Model registry lock hash does not match its contents")
    matches = [row for row in payload["resolved_models"] if row["model_key"] == model_key]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one resolved model for {model_key!r}")
    return ResolvedModel.model_validate(matches[0])
