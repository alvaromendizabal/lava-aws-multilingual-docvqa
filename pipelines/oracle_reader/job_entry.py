"""SageMaker entry point for a bounded oracle-evidence Qwen3.5 benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lava.readers.benchmark import run_oracle_benchmark
from lava.readers.model_registry import load_resolved_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--manifest-s3-uri",
        "--manifest_s3_uri",
        dest="manifest_s3_uri",
        required=True,
    )
    parser.add_argument(
        "--output-s3-prefix",
        "--output_s3_prefix",
        dest="output_s3_prefix",
        required=True,
    )
    parser.add_argument(
        "--protocol-lock-id",
        "--protocol_lock_id",
        dest="protocol_lock_id",
        required=True,
    )
    parser.add_argument("--model-key", "--model_key", dest="model_key", required=True)
    parser.add_argument(
        "--experiment-id",
        "--experiment_id",
        dest="experiment_id",
        required=True,
    )
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    model_spec = load_resolved_model(
        ROOT / "configs/oracle_reader_models.lock.json",
        args.model_key,
    )
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ORACLE_READER_JOB_COMPLETE")


if __name__ == "__main__":
    main()
