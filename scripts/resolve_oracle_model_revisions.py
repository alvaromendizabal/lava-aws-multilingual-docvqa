"""Extend the canonical oracle-reader model lock safely."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from lava.readers.model_registry import resolve_registry
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> None:
    """Resolve only appended candidates with visible runtime telemetry."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/oracle_reader_benchmark.yaml"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/oracle_reader_models.lock.json"),
    )

    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
    )

    args = parser.parse_args()

    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")

    started = time.perf_counter()

    logger = RuntimeEventLogger("oracle_reader.model_registry")

    logger.emit(
        "resolver.started",
        config=str(args.config),
        output=str(args.output),
        heartbeat_seconds=(args.heartbeat_seconds),
    )

    def progress(
        event: str,
        fields: dict[str, Any] | Any,
    ) -> None:
        payload = dict(fields)

        event_name = event if event.startswith("registry.") else f"registry.{event}"

        logger.emit(
            event_name,
            **payload,
        )

    try:
        with logger.stage(
            "registry.resolve",
            heartbeat_seconds=(args.heartbeat_seconds),
        ):
            lock = resolve_registry(
                args.config,
                args.output,
                progress=progress,
            )

    except Exception as error:
        logger.emit(
            "resolver.failed",
            level="ERROR",
            exception_type=(type(error).__name__),
            total_elapsed_seconds=round(
                time.perf_counter() - started,
                3,
            ),
        )
        raise

    total_elapsed = round(
        time.perf_counter() - started,
        3,
    )

    logger.emit(
        "resolver.completed",
        schema_version=lock["schema_version"],
        candidate_count=lock["candidate_count"],
        unique_model_repository_count=lock["unique_model_repository_count"],
        registry_sha256=lock["registry_sha256"],
        parent_registry_sha256=lock.get("parent_registry_sha256"),
        lineage_depth=lock.get(
            "lineage_depth",
            0,
        ),
        appended_model_keys=lock.get(
            "appended_model_keys",
            [],
        ),
        total_elapsed_seconds=(total_elapsed),
    )

    print(f"MODEL_REGISTRY_SHA256={lock['registry_sha256']}")

    print(f"MODEL_CANDIDATE_COUNT={lock['candidate_count']}")

    print(f"UNIQUE_MODEL_REPOSITORY_COUNT={lock['unique_model_repository_count']}")

    print(f"MODEL_REGISTRY_SCHEMA_VERSION={lock['schema_version']}")

    print(f"MODEL_REGISTRY_LINEAGE_DEPTH={lock.get('lineage_depth', 0)}")

    print("MODEL_REVISIONS_LOCKED")


if __name__ == "__main__":
    main()
