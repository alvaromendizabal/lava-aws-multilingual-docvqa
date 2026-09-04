from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

import lava.readers.model_registry as registry
from lava.readers.model_registry import (
    load_registry_lock,
    resolve_registry,
)
from lava.readers.schemas import (
    DecodingMode,
    GenerationSpec,
    ModelCandidate,
    ReaderInputMode,
    ResolvedModel,
)


def _candidate(
    model_key: str,
    model_id: str,
    *,
    input_mode: ReaderInputMode = ReaderInputMode.FUSED,
) -> ModelCandidate:
    return ModelCandidate(
        model_key=model_key,
        model_id=model_id,
        expected_license="apache-2.0",
        expected_pipeline_tag="image-text-to-text",
        parameters_billion=4.0,
        instance_type="ml.g5.2xlarge",
        input_mode=input_mode,
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


def _candidate_values(
    model_id: str,
    *,
    input_mode: str = "fused",
    instance_type: str = "ml.g5.2xlarge",
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "expected_license": "apache-2.0",
        "expected_pipeline_tag": "image-text-to-text",
        "parameters_billion": 4.0,
        "instance_type": instance_type,
        "input_mode": input_mode,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "use_kernels": False,
        "processor_min_pixels": 200704,
        "processor_max_pixels": 1605632,
        "generation": {
            "mode": "direct",
            "max_new_tokens": 64,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "seed": 1,
        },
    }


def _write_config(
    path: Path,
    models: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": dict(models),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _legacy_row() -> dict[str, Any]:
    candidate = _candidate(
        "existing",
        "Org/Existing",
    )

    model = ResolvedModel(
        **candidate.model_dump(),
        revision="a" * 40,
        observed_license="apache-2.0",
        observed_pipeline_tag="image-text-to-text",
        resolved_at_utc="2026-09-02T00:00:00+00:00",
        last_modified=None,
        gated=False,
        private=False,
    )

    row = model.model_dump(mode="json")

    for field in (
        "reader_family",
        "device_placement",
        "min_cuda_devices",
        "min_cuda_memory_per_device_gib",
        "min_total_cuda_memory_gib",
        "trust_remote_code",
    ):
        row.pop(
            field,
            None,
        )

    return row


def _write_v2_lock(
    lock_path: Path,
    config_path: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": 2,
        "generated_at_utc": ("2026-09-02T00:00:00+00:00"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidate_count": 1,
        "unique_model_repository_count": 1,
        "resolved_models": [
            row,
        ],
    }

    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode()
    ).hexdigest()

    payload = {
        **body,
        "registry_sha256": digest,
    }

    lock_path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return payload


def _metadata(
    revision: str,
) -> dict[str, Any]:
    return {
        "sha": revision,
        "pipeline_tag": "image-text-to-text",
        "cardData": {
            "license": "apache-2.0",
        },
        "private": False,
        "gated": False,
        "lastModified": ("2026-09-03T00:00:00Z"),
    }


def test_noop_preserves_lock_bytes_and_never_fetches(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    before = lock.read_bytes()

    result = resolve_registry(
        config,
        lock,
        fetcher=lambda _: pytest.fail("No-op extension must not access the network"),
    )

    assert lock.read_bytes() == before
    assert result["schema_version"] == 2


def test_extension_preserves_parent_row_exactly_and_records_ancestry(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    parent_row = _legacy_row()

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    parent = _write_v2_lock(
        lock,
        config,
        parent_row,
    )

    parent_file_sha = hashlib.sha256(lock.read_bytes()).hexdigest()

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
            "new": _candidate_values(
                "Org/New",
                instance_type=("ml.p6-b300.48xlarge"),
            ),
        },
    )

    events: list[
        tuple[
            str,
            dict[str, Any],
        ]
    ] = []

    result = resolve_registry(
        config,
        lock,
        fetcher=lambda _: _metadata("b" * 40),
        progress=lambda event, fields: events.append(
            (
                event,
                dict(fields),
            )
        ),
    )

    assert result["schema_version"] == 3
    assert result["parent_registry_sha256"] == parent["registry_sha256"]
    assert result["parent_file_sha256"] == parent_file_sha
    assert result["lineage_depth"] == 1
    assert result["preserved_model_keys"] == ["existing"]
    assert result["appended_model_keys"] == ["new"]
    assert result["resolved_models"][0] == parent_row
    assert result["resolved_models"][1]["revision"] == ("b" * 40)
    assert any(event == "registry.extended" for event, _ in events)

    validated = load_registry_lock(lock)

    assert validated["registry_sha256"] == result["registry_sha256"]


def test_frozen_candidate_cannot_drift(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    before = lock.read_bytes()

    _write_config(
        config,
        {
            "existing": _candidate_values(
                "Org/Existing",
                instance_type=("ml.g7e.48xlarge"),
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="candidate drift",
    ):
        resolve_registry(
            config,
            lock,
            fetcher=lambda _: pytest.fail("Drift must fail before network access"),
        )

    assert lock.read_bytes() == before


def test_frozen_model_key_cannot_be_removed(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    before = lock.read_bytes()

    _write_config(
        config,
        {
            "different": _candidate_values("Org/Different"),
        },
    )

    with pytest.raises(
        ValueError,
        match="cannot remove frozen model keys",
    ):
        resolve_registry(
            config,
            lock,
        )

    assert lock.read_bytes() == before


def test_new_variant_of_frozen_repository_reuses_revision_without_network(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
            "image_variant": _candidate_values(
                "Org/Existing",
                input_mode="image_only",
            ),
        },
    )

    result = resolve_registry(
        config,
        lock,
        fetcher=lambda _: pytest.fail("Frozen repository must not be re-resolved"),
    )

    appended = result["resolved_models"][1]

    assert appended["revision"] == "a" * 40
    assert appended["model_key"] == "image_variant"


def test_retryable_metadata_failure_is_retried_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
            "new": _candidate_values("Org/New"),
        },
    )

    monkeypatch.setattr(
        registry.time,
        "sleep",
        lambda _: None,
    )

    calls = 0
    events: list[str] = []

    def fetcher(
        _: str,
    ) -> Mapping[str, Any]:
        nonlocal calls

        calls += 1

        if calls < 3:
            raise requests.ConnectionError("temporary connectivity failure")

        return _metadata("b" * 40)

    result = resolve_registry(
        config,
        lock,
        fetcher=fetcher,
        progress=lambda event, _: events.append(event),
    )

    assert calls == 3
    assert events.count("model.metadata_fetch.retry") == 2
    assert result["resolved_models"][1]["revision"] == "b" * 40


def test_failed_new_repository_resolution_leaves_parent_lock_untouched(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    before = lock.read_bytes()

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
            "new": _candidate_values("Org/New"),
        },
    )

    response = requests.Response()
    response.status_code = 404

    error = requests.HTTPError(
        "not found",
        response=response,
    )

    def fail(
        _: str,
    ) -> Mapping[str, Any]:
        raise error

    with pytest.raises(
        requests.HTTPError,
    ):
        resolve_registry(
            config,
            lock,
            fetcher=fail,
        )

    assert lock.read_bytes() == before


def test_atomic_replace_failure_preserves_parent_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "lock.json"

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
        },
    )

    _write_v2_lock(
        lock,
        config,
        _legacy_row(),
    )

    before = lock.read_bytes()

    _write_config(
        config,
        {
            "existing": _candidate_values("Org/Existing"),
            "new": _candidate_values("Org/New"),
        },
    )

    def fail_replace(
        *_: object,
    ) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(
        registry.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="atomic replace",
    ):
        resolve_registry(
            config,
            lock,
            fetcher=lambda _: _metadata("b" * 40),
        )

    assert lock.read_bytes() == before
    assert not list(tmp_path.glob(".lock.json.*.tmp"))


def test_resolver_script_uses_structured_utc_logging_and_heartbeat() -> None:
    root = Path(__file__).resolve().parents[2]

    source = (root / "scripts" / "resolve_oracle_model_revisions.py").read_text(encoding="utf-8")

    assert "RuntimeEventLogger" in source
    assert '"oracle_reader.model_registry"' in source
    assert "heartbeat_seconds" in source
    assert '"resolver.started"' in source
    assert '"resolver.completed"' in source
    assert '"resolver.failed"' in source
    assert '"registry.resolve"' in source
    assert 'event.startswith("registry.")' in source
    assert 'else f"registry.{event}"' in source
