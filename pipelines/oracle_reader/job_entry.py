"""SageMaker entry point for a bounded oracle-evidence Qwen3.5 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lava.readers.benchmark import run_oracle_benchmark
from lava.readers.model_registry import load_resolved_model
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> None:
    """Run the bounded benchmark with visible UTC stages and heartbeats."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--manifest-s3-uri", "--manifest_s3_uri", dest="manifest_s3_uri", required=True
    )
    parser.add_argument(
        "--output-s3-prefix", "--output_s3_prefix", dest="output_s3_prefix", required=True
    )
    parser.add_argument(
        "--protocol-lock-id", "--protocol_lock_id", dest="protocol_lock_id", required=True
    )
    parser.add_argument("--model-key", "--model_key", dest="model_key", required=True)
    parser.add_argument("--experiment-id", "--experiment_id", dest="experiment_id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger = RuntimeEventLogger("oracle_reader.job")
    logger.emit(
        "job.started",
        experiment_id=args.experiment_id,
        model_key=args.model_key,
        limit=args.limit,
        protocol_lock_id=args.protocol_lock_id,
    )
    with logger.stage("model_spec", heartbeat_seconds=15.0):
        model_spec = load_resolved_model(
            ROOT / "configs/oracle_reader_models.lock.json", args.model_key
        )
    with logger.stage("benchmark", heartbeat_seconds=15.0):
        summary = run_oracle_benchmark(
            bucket=args.bucket,
            region=args.region,
            manifest_s3_uri=args.manifest_s3_uri,
            output_s3_prefix=args.output_s3_prefix,
            protocol_lock_id=args.protocol_lock_id,
            model_spec=model_spec,
            experiment_id=args.experiment_id,
            limit=args.limit,
        )
    logger.emit(
        "job.summary",
        record_count=summary.get("record_count"),
        schema_valid_rate=summary.get("schema_valid_rate"),
        parser_error_counts=summary.get("parser_error_counts"),
        abstention_rate=summary.get("abstention_rate"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    logger.emit("job.completed")
    print("ORACLE_READER_JOB_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
