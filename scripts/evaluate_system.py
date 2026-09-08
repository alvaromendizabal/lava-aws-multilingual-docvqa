"""Complete the frozen retrieved-page pilot; default preview never allocates compute."""

from __future__ import annotations

import argparse
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from lava.evaluation.access import check_cpu_memory, check_judge_access
from lava.evaluation.semantic import judge_contract
from lava.evaluation.system import evaluate_system, load_summary
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import prepare_inputs, put_blob, store_for, system_contract
from lava.readers.system_execution import SYSTEM_INSTANCES, execute_system


def main() -> int:
    """One recoverable operator command, with a separate and explicit new-compute gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preview", "prepare", "run", "evaluate", "finish", "notebook"),
        default="preview",
    )
    parser.add_argument("--acknowledge-charges", choices=("YES", "NO"), default="NO")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry", choices=("YES", "NO"), default="NO")
    parser.add_argument("--hourly-usd-ceiling", type=float, default=5.0)
    parser.add_argument("--maximum-training-usd", type=float, default=5.0)
    parser.add_argument(
        "--instance-type", choices=sorted(SYSTEM_INSTANCES), default="ml.g6e.2xlarge"
    )
    args = parser.parse_args()
    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    path = root / "artifacts/system/runtime" / run_id / "events.jsonl"
    logger = RuntimeEventLogger("system.command", jsonl_path=path)
    contract = system_contract(root)
    logger.emit(
        "system.plan",
        mode=args.mode,
        contract_id=contract["contract_id"],
        model=contract["model"]["model_id"],
        page_budget=5,
        question_count=16,
        gpu_runtime_cap_seconds=1800,
        per_attempt_training_budget=args.maximum_training_usd,
        kaggle_submission=False,
    )
    if args.mode == "preview":
        summary = load_summary(root)
        logger.emit(
            "system.preview.completed",
            paid_resource_created=False,
            current_result="scored" if summary else "not_yet_measured",
        )
        return 0
    if args.mode in {"run", "finish", "notebook"} and args.acknowledge_charges != "YES":
        raise ValueError(
            "Run/finish requires CHARGES=YES; use preview or evaluate for no new GPU work"
        )
    load_dotenv(root / ".env", override=False)
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-west-2")
    if not bucket:
        raise ValueError("S3_BUCKET is missing from the existing project .env")
    session = boto3.Session(region_name=region)
    s3 = session.client(
        "s3",
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
        ),
    )
    try:
        with logger.stage("system.finish", heartbeat_seconds=15):
            if args.mode in {"finish", "notebook"}:
                # Detect judge login/memory problems before any new paid GPU work.
                check_cpu_memory(logger)
                check_judge_access(judge_contract(root)["config"], logger)
            if args.mode in {"prepare", "finish", "notebook"}:
                prepare_inputs(root, s3, bucket, logger)
            if args.mode in {"run", "finish", "notebook"}:
                execute_system(
                    root,
                    session,
                    bucket,
                    region,
                    logger,
                    acknowledge_charges=args.acknowledge_charges,
                    attempt=args.attempt,
                    allow_retry=args.retry == "YES",
                    hourly_usd_ceiling=args.hourly_usd_ceiling,
                    maximum_training_usd=args.maximum_training_usd,
                    instance_type=args.instance_type,
                )
            if args.mode in {"evaluate", "finish", "notebook"}:
                summary = evaluate_system(root, s3, bucket, logger)
                logger.emit("system.results", **summary["metrics"]["question_micro"])
            if args.mode == "finish":
                for command in (
                    [
                        "uv",
                        "run",
                        "--frozen",
                        "python",
                        "-m",
                        "ipykernel",
                        "install",
                        "--user",
                        "--name",
                        "lava",
                        "--display-name",
                        "LAVA",
                    ],
                    ["make", "notebooks"],
                    ["make", "quality"],
                ):
                    subprocess.run(command, cwd=root, check=True)
                # Archive exact successful notebook bytes and manifests outside the host.
                prefix = f"experiments/submissions/system/{contract['contract_id']}/publication"
                publication = {}
                for folder, pattern in (
                    ("notebooks", "*.ipynb"),
                    ("reports/notebook_execution", "*.json"),
                ):
                    for notebook in sorted((root / folder).glob(pattern)):
                        from lava.evaluation.semantic import digest

                        payload = notebook.read_bytes()
                        relative = str(notebook.relative_to(root))
                        key = f"{prefix}/{digest(payload)}/{notebook.name}"
                        put_blob(s3, bucket, key, payload, "application/json")
                        publication[relative] = {"key": key, "sha256": digest(payload)}
                store = store_for(s3, bucket, contract["contract_id"])
                from lava.evaluation.semantic import digest, encode

                key = f"publication/{digest(encode(publication))}.json"
                store.write(key, publication)
                if store.read(key) != publication:
                    raise ValueError("Notebook publication failed durable read-back")
                logger.emit(
                    "system.portfolio.ready",
                    notebooks=6,
                    quality_gate="passed",
                    kaggle_submission=False,
                )
        return 0
    except KeyboardInterrupt:
        logger.emit(
            "system.interrupted",
            action="Repeat the same command; accepted jobs and answers remain saved",
        )
        return 130
    finally:
        # Preserve diagnostics even after a model/judge failure; no raw answers enter this log.
        if path.exists():
            try:
                put_blob(
                    s3,
                    bucket,
                    f"experiments/submissions/system/{contract['contract_id']}/logs/{run_id}.jsonl",
                    path.read_bytes(),
                    "application/x-ndjson",
                )
            except (BotoCoreError, ClientError, OSError, ValueError) as error:
                # Keep the original exception, not an archival failure, as the primary result.
                logger.emit(
                    "system.log.archive_failed",
                    level="ERROR",
                    error_type=type(error).__name__,
                    local_path=str(path),
                )


if __name__ == "__main__":
    raise SystemExit(main())
