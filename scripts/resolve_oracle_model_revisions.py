"""Resolve configured reader candidates to immutable Hugging Face commits."""

from __future__ import annotations

import argparse
from pathlib import Path

from lava.readers.model_registry import resolve_registry


def main() -> None:
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
    args = parser.parse_args()
    lock = resolve_registry(args.config, args.output)
    print(f"MODEL_REGISTRY_SHA256={lock['registry_sha256']}")
    print(f"MODEL_CANDIDATE_COUNT={lock['candidate_count']}")
    print(f"UNIQUE_MODEL_REPOSITORY_COUNT={lock['unique_model_repository_count']}")
    print("MODEL_REVISIONS_LOCKED")


if __name__ == "__main__":
    main()
