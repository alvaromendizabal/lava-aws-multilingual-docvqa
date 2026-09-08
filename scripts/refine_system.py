"""Run the frozen self-citation refinement; preview never creates compute."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime

import boto3
from dotenv import load_dotenv

from lava.evaluation.access import check_cpu_memory, check_judge_access
from lava.evaluation.semantic import judge_contract
from lava.notebook_support import find_repo_root
from lava.readers.refinement import (
    evaluate_refinement,
    prepare_refinement,
    refinement_contract,
    refinement_training_request,
)
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import put_blob
from lava.readers.system_execution import execute_prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preview", "prepare", "finish", "evaluate"), default="preview"
    )
    parser.add_argument("--acknowledge-charges", choices=("YES", "NO"), default="NO")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry", choices=("YES", "NO"), default="NO")
    args = parser.parse_args()
    root = find_repo_root()
    contract = refinement_contract(root)
    attempt = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    log = root / "artifacts/system/refinement" / f"{attempt}.jsonl"
    logger = RuntimeEventLogger("refinement.command", jsonl_path=log)
    logger.emit(
        "refinement.plan",
        contract_id=contract["contract_id"],
        policy="reread_self_cited_pages",
        mode=args.mode,
        questions=16,
        maximum_training_usd=5,
    )
    if args.mode == "preview":
        return 0
    if args.mode == "finish" and args.acknowledge_charges != "YES":
        raise ValueError("New GPU work requires explicit charge acknowledgement")
    load_dotenv(root / ".env", override=False)
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-west-2")
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    try:
        if args.mode in {"prepare", "finish"}:
            if args.mode == "finish":
                check_cpu_memory(logger)
                check_judge_access(judge_contract(root)["config"], logger)
            prepare_refinement(root, s3, bucket, logger)
        if args.mode == "finish":
            execute_prepared(
                root,
                session,
                bucket,
                region,
                logger,
                contract=contract,
                request_factory=refinement_training_request,
                receipt_directory="reports/system/refinement",
                acknowledge_charges=args.acknowledge_charges,
                attempt=args.attempt,
                allow_retry=args.retry == "YES",
                hourly_usd_ceiling=6,
                maximum_training_usd=5,
                instance_type="ml.g6e.8xlarge",
            )
        if args.mode in {"finish", "evaluate"}:
            evaluate_refinement(root, s3, bucket, logger)
    finally:
        put_blob(
            s3,
            bucket,
            f"experiments/submissions/system/{contract['contract_id']}/logs/{attempt}.jsonl",
            log.read_bytes(),
            "application/x-ndjson",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
