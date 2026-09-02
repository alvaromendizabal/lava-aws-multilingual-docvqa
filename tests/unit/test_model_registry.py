import hashlib
import json
from pathlib import Path

import pytest

from lava.readers.model_registry import (
    load_resolved_model,
    resolve_candidate,
)
from lava.readers.schemas import (
    DecodingMode,
    GenerationSpec,
    ModelCandidate,
    ReaderInputMode,
)


def candidate() -> ModelCandidate:
    return ModelCandidate(
        model_key="qwen",
        model_id="Qwen/Qwen3.5-4B",
        expected_license="apache-2.0",
        expected_pipeline_tag="image-text-to-text",
        parameters_billion=4,
        instance_type="ml.g5.2xlarge",
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
    )


def test_model_registry_locks_revision_and_validates_metadata() -> None:
    resolved = resolve_candidate(
        candidate(),
        fetcher=lambda _: {
            "sha": "a" * 40,
            "pipeline_tag": "image-text-to-text",
            "cardData": {"license": "apache-2.0"},
            "private": False,
            "gated": False,
        },
        resolved_at_utc="2026-09-02T00:00:00+00:00",
    )
    assert resolved.revision == "a" * 40
    assert resolved.observed_license == "apache-2.0"


def test_model_registry_rejects_gated_models() -> None:
    with pytest.raises(ValueError, match="public and ungated"):
        resolve_candidate(
            candidate(),
            fetcher=lambda _: {
                "sha": "a" * 40,
                "pipeline_tag": "image-text-to-text",
                "cardData": {"license": "apache-2.0"},
                "private": False,
                "gated": True,
            },
        )


def test_load_resolved_model_detects_tampered_lock(tmp_path: Path) -> None:
    resolved = resolve_candidate(
        candidate(),
        fetcher=lambda _: {
            "sha": "a" * 40,
            "pipeline_tag": "image-text-to-text",
            "cardData": {"license": "apache-2.0"},
        },
        resolved_at_utc="2026-09-02T00:00:00+00:00",
    )
    body = {
        "schema_version": 2,
        "generated_at_utc": "2026-09-02T00:00:00+00:00",
        "config_sha256": "b" * 64,
        "candidate_count": 1,
        "unique_model_repository_count": 1,
        "resolved_models": [resolved.model_dump(mode="json")],
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload = {**body, "registry_sha256": hashlib.sha256(canonical).hexdigest()}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload))
    assert load_resolved_model(path, "qwen").revision == "a" * 40
    payload["candidate_count"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        load_resolved_model(path, "qwen")
