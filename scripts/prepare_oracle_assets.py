"""Build private gold-page assets and sanitized public summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from lava.readers.oracle_assets import prepare_oracle_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/oracle_reader_benchmark.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/oracle_reader"),
    )
    args = parser.parse_args()
    if not args.bucket:
        raise SystemExit("S3_BUCKET or --bucket is required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = prepare_oracle_assets(
        bucket=args.bucket,
        region=args.region,
        config=config,
        output_dir=args.output_dir,
    )
    print(f"ORACLE_QUESTION_COUNT={summary['question_count']}")
    print(f"ORACLE_EVIDENCE_PAGE_COUNT={summary['unique_evidence_page_count']}")
    print(f"ORACLE_MANIFEST_SHA256={summary['private_manifest_sha256']}")
    print("ORACLE_ASSETS_VERIFIED")


if __name__ == "__main__":
    main()
