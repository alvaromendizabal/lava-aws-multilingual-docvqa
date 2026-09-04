"""Resolve and extend immutable Hugging Face model revision locks.

Existing registry rows are append-only: adding new model candidates can
never re-resolve, rewrite, remove, or mutate an already-frozen candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
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
ProgressCallback = Callable[[str, Mapping[str, Any]], None]

_MAX_FETCH_ATTEMPTS = 4
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 8.0


def load_candidates(path: Path) -> tuple[ModelCandidate, ...]:
    """Load and strictly validate reader candidates from benchmark YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict) or not isinstance(
        payload.get("models"),
        dict,
    ):
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
            repetition_penalty=float(
                generation_values.get(
                    "repetition_penalty",
                    1.0,
                )
            ),
            seed=int(generation_values["seed"]),
        )

        candidates.append(
            ModelCandidate(
                model_key=str(model_key),
                model_id=values["model_id"],
                expected_license=values["expected_license"],
                expected_pipeline_tag=values["expected_pipeline_tag"],
                parameters_billion=float(values["parameters_billion"]),
                reader_family=ReaderFamily(
                    values.get(
                        "reader_family",
                        ReaderFamily.QWEN3_5.value,
                    )
                ),
                device_placement=DevicePlacement(
                    values.get(
                        "device_placement",
                        DevicePlacement.SINGLE.value,
                    )
                ),
                min_cuda_devices=int(
                    values.get(
                        "min_cuda_devices",
                        1,
                    )
                ),
                min_cuda_memory_per_device_gib=int(
                    values.get(
                        "min_cuda_memory_per_device_gib",
                        1,
                    )
                ),
                min_total_cuda_memory_gib=int(
                    values.get(
                        "min_total_cuda_memory_gib",
                        1,
                    )
                ),
                trust_remote_code=bool(
                    values.get(
                        "trust_remote_code",
                        False,
                    )
                ),
                instance_type=values["instance_type"],
                input_mode=ReaderInputMode(values["input_mode"]),
                dtype=values["dtype"],
                attention_implementation=values["attention_implementation"],
                use_kernels=bool(
                    values.get(
                        "use_kernels",
                        False,
                    )
                ),
                processor_min_pixels=int(values["processor_min_pixels"]),
                processor_max_pixels=int(values["processor_max_pixels"]),
                generation=generation,
            )
        )

    if len(candidates) != len({item.model_key for item in candidates}):
        raise ValueError("Model keys must be unique")

    return tuple(candidates)


def _default_fetcher(
    model_id: str,
) -> Mapping[str, Any]:
    """Fetch one Hugging Face repository snapshot without hidden retries."""
    with requests.Session() as session:
        response = session.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=30,
            headers={
                "Accept": "application/json",
                "User-Agent": ("lava-oracle-model-registry/1.0"),
            },
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
    """Validate repository metadata and lock one candidate to a Git SHA."""
    payload = metadata if metadata is not None else fetcher(candidate.model_id)

    revision = str(
        payload.get(
            "sha",
            "",
        )
    ).lower()

    card_data = payload.get("cardData") or {}

    if not isinstance(
        card_data,
        Mapping,
    ):
        card_data = {}

    observed_license = str(
        card_data.get(
            "license",
            payload.get(
                "license",
                "",
            ),
        )
    ).casefold()

    observed_pipeline_tag = str(
        payload.get(
            "pipeline_tag",
            "",
        )
    )

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
        observed_pipeline_tag=(observed_pipeline_tag),
        resolved_at_utc=(resolved_at_utc or datetime.now(UTC).isoformat()),
        last_modified=(str(payload["lastModified"]) if payload.get("lastModified") else None),
        gated=bool(
            payload.get(
                "gated",
                False,
            )
        ),
        private=bool(
            payload.get(
                "private",
                False,
            )
        ),
    )


def _canonical_registry_body(
    body: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()


def _file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_lower_sha256(
    value: object,
) -> bool:
    return (
        isinstance(
            value,
            str,
        )
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_lock(
    body: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **body,
        "registry_sha256": hashlib.sha256(_canonical_registry_body(body)).hexdigest(),
    }


def _candidate_contract(
    model: ResolvedModel,
) -> dict[str, Any]:
    """Project a resolved row back to its immutable candidate fields."""
    dumped = model.model_dump(mode="json")

    return {field_name: dumped[field_name] for field_name in ModelCandidate.model_fields}


def _emit(
    progress: ProgressCallback | None,
    event: str,
    **fields: Any,
) -> None:
    if progress is not None:
        progress(
            event,
            fields,
        )


def _retryable_request_error(
    error: requests.RequestException,
) -> bool:
    if isinstance(
        error,
        (
            requests.ConnectionError,
            requests.Timeout,
        ),
    ):
        return True

    if isinstance(
        error,
        requests.HTTPError,
    ):
        response = error.response

        if response is None:
            return False

        return response.status_code in {
            408,
            429,
            500,
            502,
            503,
            504,
        }

    return False


def _fetch_with_retry(
    candidate: ModelCandidate,
    *,
    fetcher: JsonFetcher,
    progress: ProgressCallback | None,
) -> Mapping[str, Any]:
    """Fetch metadata with bounded exponential retry and visible progress."""
    for attempt in range(
        1,
        _MAX_FETCH_ATTEMPTS + 1,
    ):
        _emit(
            progress,
            "model.metadata_fetch.started",
            model_key=candidate.model_key,
            model_id=candidate.model_id,
            attempt=attempt,
            max_attempts=_MAX_FETCH_ATTEMPTS,
        )

        try:
            payload = fetcher(candidate.model_id)

        except requests.RequestException as error:
            retryable = _retryable_request_error(error)

            if not retryable or attempt >= _MAX_FETCH_ATTEMPTS:
                _emit(
                    progress,
                    "model.metadata_fetch.failed",
                    model_key=candidate.model_key,
                    model_id=candidate.model_id,
                    attempt=attempt,
                    retryable=retryable,
                    exception_type=(type(error).__name__),
                )
                raise

            delay = min(
                _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                _MAX_BACKOFF_SECONDS,
            )

            _emit(
                progress,
                "model.metadata_fetch.retry",
                model_key=candidate.model_key,
                model_id=candidate.model_id,
                attempt=attempt,
                next_attempt=attempt + 1,
                backoff_seconds=delay,
                exception_type=(type(error).__name__),
            )

            time.sleep(delay)

        else:
            _emit(
                progress,
                "model.metadata_fetch.completed",
                model_key=candidate.model_key,
                model_id=candidate.model_id,
                attempt=attempt,
            )

            return payload

    raise RuntimeError("Metadata retry loop exited unexpectedly")


def load_registry_lock(
    path: Path,
) -> dict[str, Any]:
    """Load and fully validate a schema-v2 or schema-v3 registry lock."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError("Model registry lock must be a JSON object")

    schema_version = payload.get("schema_version")

    if schema_version not in {
        2,
        3,
    }:
        raise ValueError(f"Unsupported model registry schema version: {schema_version!r}")

    expected = payload.get("registry_sha256")

    body = {key: value for key, value in payload.items() if key != "registry_sha256"}

    observed = hashlib.sha256(_canonical_registry_body(body)).hexdigest()

    if expected != observed:
        raise ValueError("Model registry lock hash does not match its contents")

    if not _is_lower_sha256(payload.get("config_sha256")):
        raise ValueError("config_sha256 must be a lowercase SHA-256")

    rows = payload.get("resolved_models")

    if not isinstance(
        rows,
        list,
    ):
        raise TypeError("resolved_models must be a list")

    if not all(
        isinstance(
            row,
            dict,
        )
        for row in rows
    ):
        raise TypeError("Every resolved model row must be an object")

    if payload.get("candidate_count") != len(rows):
        raise ValueError("candidate_count does not match resolved_models")

    keys = [
        str(
            row.get(
                "model_key",
                "",
            )
        )
        for row in rows
    ]

    if any(not key for key in keys):
        raise ValueError("Every resolved model must have a model_key")

    if len(keys) != len(set(keys)):
        raise ValueError("Resolved model keys must be unique")

    repository_ids = {
        str(
            row.get(
                "model_id",
                "",
            )
        )
        for row in rows
    }

    if "" in repository_ids:
        raise ValueError("Every resolved model must have a model_id")

    if payload.get("unique_model_repository_count") != len(repository_ids):
        raise ValueError("unique_model_repository_count is inconsistent")

    for row in rows:
        ResolvedModel.model_validate(row)

    if schema_version == 3:
        parent_registry = payload.get("parent_registry_sha256")

        parent_file = payload.get("parent_file_sha256")

        lineage_depth = payload.get("lineage_depth")

        preserved = payload.get("preserved_model_keys")

        appended = payload.get("appended_model_keys")

        if not isinstance(
            lineage_depth,
            int,
        ):
            raise TypeError("lineage_depth must be an integer")

        if not isinstance(
            preserved,
            list,
        ):
            raise TypeError("preserved_model_keys must be a list")

        if not isinstance(
            appended,
            list,
        ):
            raise TypeError("appended_model_keys must be a list")

        preserved_keys = [str(item) for item in preserved]

        appended_keys = [str(item) for item in appended]

        if len(preserved_keys) != len(set(preserved_keys)):
            raise ValueError("preserved_model_keys must be unique")

        if len(appended_keys) != len(set(appended_keys)):
            raise ValueError("appended_model_keys must be unique")

        if set(preserved_keys) & set(appended_keys):
            raise ValueError("Preserved and appended model keys must be disjoint")

        if set(preserved_keys) | set(appended_keys) != set(keys):
            raise ValueError("Schema-v3 lineage keys must partition all models")

        if parent_registry is None:
            if parent_file is not None or lineage_depth != 0 or preserved_keys:
                raise ValueError("Initial schema-v3 locks cannot declare a parent")
        else:
            if not _is_lower_sha256(parent_registry):
                raise ValueError("parent_registry_sha256 must be a lowercase SHA-256")

            if not _is_lower_sha256(parent_file):
                raise ValueError("parent_file_sha256 must be a lowercase SHA-256")

            if lineage_depth < 1:
                raise ValueError("Extended schema-v3 locks require positive lineage_depth")

    return payload


def _atomic_write_lock(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Durably replace the canonical lock without persistent side files."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            temporary_path.chmod(path.stat().st_mode)
        else:
            temporary_path.chmod(0o644)

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _initialize_registry(
    config_path: Path,
    output_path: Path,
    *,
    fetcher: JsonFetcher,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    candidates = load_candidates(config_path)

    timestamp = datetime.now(UTC).isoformat()

    cache: dict[
        str,
        Mapping[str, Any],
    ] = {}

    resolved: list[ResolvedModel] = []

    for candidate in candidates:
        _emit(
            progress,
            "model.resolving",
            model_key=candidate.model_key,
            model_id=candidate.model_id,
            resolution_mode="new_repository",
        )

        if candidate.model_id not in cache:
            cache[candidate.model_id] = _fetch_with_retry(
                candidate,
                fetcher=fetcher,
                progress=progress,
            )

        model = resolve_candidate(
            candidate,
            resolved_at_utc=timestamp,
            metadata=cache[candidate.model_id],
        )

        resolved.append(model)

        _emit(
            progress,
            "model.resolved",
            model_key=model.model_key,
            model_id=model.model_id,
            revision=model.revision,
        )

    rows = [model.model_dump(mode="json") for model in resolved]

    body = {
        "schema_version": 3,
        "generated_at_utc": timestamp,
        "config_sha256": _file_sha256(config_path),
        "candidate_count": len(rows),
        "unique_model_repository_count": len({row["model_id"] for row in rows}),
        "parent_registry_sha256": None,
        "parent_file_sha256": None,
        "lineage_depth": 0,
        "preserved_model_keys": [],
        "appended_model_keys": [row["model_key"] for row in rows],
        "resolved_models": rows,
    }

    lock = _build_lock(body)

    _atomic_write_lock(
        output_path,
        lock,
    )

    validated = load_registry_lock(output_path)

    _emit(
        progress,
        "registry.initialized",
        candidate_count=len(rows),
        registry_sha256=validated["registry_sha256"],
    )

    return validated


def _extend_registry(
    config_path: Path,
    output_path: Path,
    *,
    fetcher: JsonFetcher,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    parent = load_registry_lock(output_path)

    parent_file_sha = _file_sha256(output_path)

    candidates = load_candidates(config_path)

    candidates_by_key = {candidate.model_key: candidate for candidate in candidates}

    parent_rows = parent["resolved_models"]

    parent_keys = [str(row["model_key"]) for row in parent_rows]

    missing_keys = sorted(set(parent_keys) - set(candidates_by_key))

    if missing_keys:
        raise ValueError(f"Configuration cannot remove frozen model keys: {missing_keys}")

    parent_models: list[ResolvedModel] = []

    for raw_row in parent_rows:
        model = ResolvedModel.model_validate(raw_row)

        candidate = candidates_by_key[model.model_key]

        expected_contract = candidate.model_dump(mode="json")

        frozen_contract = _candidate_contract(model)

        if expected_contract != frozen_contract:
            raise ValueError(f"Frozen model candidate drift detected for {model.model_key!r}")

        parent_models.append(model)

        _emit(
            progress,
            "model.preserved",
            model_key=model.model_key,
            model_id=model.model_id,
            revision=model.revision,
        )

    parent_key_set = set(parent_keys)

    new_candidates = [
        candidate for candidate in candidates if candidate.model_key not in parent_key_set
    ]

    current_config_sha = _file_sha256(config_path)

    if not new_candidates:
        if parent.get("config_sha256") != current_config_sha:
            raise ValueError("Configuration bytes changed without appending a new model candidate")

        _emit(
            progress,
            "registry.noop",
            candidate_count=parent["candidate_count"],
            registry_sha256=parent["registry_sha256"],
        )

        return parent

    timestamp = datetime.now(UTC).isoformat()

    repository_models: dict[
        str,
        list[ResolvedModel],
    ] = {}

    for model in parent_models:
        repository_models.setdefault(
            model.model_id,
            [],
        ).append(model)

    metadata_cache: dict[
        str,
        Mapping[str, Any],
    ] = {}

    new_models: list[ResolvedModel] = []

    for candidate in new_candidates:
        frozen_repository_models = repository_models.get(
            candidate.model_id,
            [],
        )

        if frozen_repository_models:
            revisions = {model.revision for model in frozen_repository_models}

            if len(revisions) != 1:
                raise ValueError(
                    f"One frozen repository contains multiple revisions: {candidate.model_id!r}"
                )

            licenses = {model.observed_license.casefold() for model in frozen_repository_models}

            pipeline_tags = {model.observed_pipeline_tag for model in frozen_repository_models}

            if len(licenses) != 1:
                raise ValueError("Frozen repository contains inconsistent licenses")

            if len(pipeline_tags) != 1:
                raise ValueError("Frozen repository contains inconsistent pipeline tags")

            observed_license = next(iter(licenses))

            observed_pipeline_tag = next(iter(pipeline_tags))

            if observed_license != candidate.expected_license.casefold():
                raise ValueError(
                    "New candidate expected license disagrees with the frozen repository metadata"
                )

            if observed_pipeline_tag != candidate.expected_pipeline_tag:
                raise ValueError(
                    "New candidate expected pipeline tag disagrees "
                    "with the frozen repository metadata"
                )

            source = frozen_repository_models[0]

            _emit(
                progress,
                "model.resolving",
                model_key=candidate.model_key,
                model_id=candidate.model_id,
                resolution_mode="reuse_frozen_repository",
                revision=source.revision,
            )

            model = ResolvedModel(
                **candidate.model_dump(),
                revision=source.revision,
                observed_license=(source.observed_license),
                observed_pipeline_tag=(source.observed_pipeline_tag),
                resolved_at_utc=timestamp,
                last_modified=(source.last_modified),
                gated=False,
                private=False,
            )

            _emit(
                progress,
                "model.frozen_revision_reused",
                model_key=model.model_key,
                model_id=model.model_id,
                revision=model.revision,
            )

        else:
            _emit(
                progress,
                "model.resolving",
                model_key=candidate.model_key,
                model_id=candidate.model_id,
                resolution_mode="new_repository",
            )

            if candidate.model_id not in metadata_cache:
                metadata_cache[candidate.model_id] = _fetch_with_retry(
                    candidate,
                    fetcher=fetcher,
                    progress=progress,
                )

            model = resolve_candidate(
                candidate,
                resolved_at_utc=timestamp,
                metadata=metadata_cache[candidate.model_id],
            )

        new_models.append(model)

        _emit(
            progress,
            "model.resolved",
            model_key=model.model_key,
            model_id=model.model_id,
            revision=model.revision,
        )

    appended_rows = [model.model_dump(mode="json") for model in new_models]

    combined_rows = [
        *parent_rows,
        *appended_rows,
    ]

    parent_depth = (
        int(
            parent.get(
                "lineage_depth",
                0,
            )
        )
        if parent.get("schema_version") == 3
        else 0
    )

    body = {
        "schema_version": 3,
        "generated_at_utc": timestamp,
        "config_sha256": current_config_sha,
        "candidate_count": len(combined_rows),
        "unique_model_repository_count": len({str(row["model_id"]) for row in combined_rows}),
        "parent_registry_sha256": parent["registry_sha256"],
        "parent_file_sha256": parent_file_sha,
        "lineage_depth": (parent_depth + 1),
        "preserved_model_keys": parent_keys,
        "appended_model_keys": [model.model_key for model in new_models],
        "resolved_models": combined_rows,
    }

    lock = _build_lock(body)

    _atomic_write_lock(
        output_path,
        lock,
    )

    validated = load_registry_lock(output_path)

    _emit(
        progress,
        "registry.extended",
        parent_registry_sha256=parent["registry_sha256"],
        parent_file_sha256=parent_file_sha,
        registry_sha256=validated["registry_sha256"],
        preserved_count=len(parent_keys),
        appended_count=len(new_models),
        candidate_count=len(combined_rows),
        lineage_depth=validated["lineage_depth"],
    )

    return validated


def resolve_registry(
    config_path: Path,
    output_path: Path,
    *,
    fetcher: JsonFetcher = _default_fetcher,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Initialize once; every subsequent registry update is append-only."""
    if output_path.exists():
        return _extend_registry(
            config_path,
            output_path,
            fetcher=fetcher,
            progress=progress,
        )

    return _initialize_registry(
        config_path,
        output_path,
        fetcher=fetcher,
        progress=progress,
    )


def load_resolved_model(
    path: Path,
    model_key: str,
) -> ResolvedModel:
    """Load one model after validating complete registry integrity."""
    payload = load_registry_lock(path)

    matches = [row for row in payload["resolved_models"] if row["model_key"] == model_key]

    if len(matches) != 1:
        raise KeyError(f"Expected exactly one resolved model for {model_key!r}")

    return ResolvedModel.model_validate(matches[0])
