"""Score saved complete pilots; reader inference and new cloud compute are never launched."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from lava.evaluation.access import EvaluationAccessError, check_cpu_memory, check_judge_access
from lava.evaluation.reporting import load_report, write_report
from lava.evaluation.saved_predictions import (
    evaluate_semantic,
    load_saved_run,
    save_public_evaluation,
    supporting_metrics,
)
from lava.evaluation.semantic import (
    DurableSemanticJudge,
    GemmaDecision,
    ImmutableS3Objects,
    judge_contract,
)
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> int:
    """Preview scoring, calculate diagnostics, or run the resumable pinned Gemma judge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", action="append")
    parser.add_argument(
        "--mode", choices=("preview", "check", "diagnostics", "semantic"), default="preview"
    )
    args = parser.parse_args()
    root = find_repo_root(Path(__file__).resolve())
    load_dotenv(root / ".env", override=False)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    log_path = root / "artifacts/semantic_judge/runtime" / run_id / "events.jsonl"
    logger = RuntimeEventLogger("oracle_reader.evaluation", jsonl_path=log_path)
    logger.emit("evaluation.log", path=str(log_path), mode=args.mode)
    try:
        return run_evaluation(root, args, logger)
    except EvaluationAccessError as error:
        logger.emit("evaluation.action_required", level="ERROR", instruction=str(error))
        print("EVALUATION_ACTION_REQUIRED")
        return 2
    except KeyboardInterrupt:
        logger.emit(
            "evaluation.interrupted", instruction="Rerun make evaluate to reuse saved decisions."
        )
        return 130


def run_evaluation(root: Path, args: argparse.Namespace, logger: RuntimeEventLogger) -> int:
    """Run one explicitly selected operation; access checks create no cloud clients."""
    with logger.stage("evaluation", heartbeat_seconds=15):
        if args.mode == "check":
            check_cpu_memory(logger)
            check_judge_access(judge_contract(root)["config"], logger)
            print("SEMANTIC_JUDGE_ACCESS_VERIFIED")
            return 0
        report = load_report(root)
        jobs = args.job_name or [r["job_name"] for r in report["current_models"] if r["complete"]]
        if not jobs:
            raise ValueError("No complete verified pilots are available")
        contract = judge_contract(root)
        logger.emit(
            "evaluation.plan",
            mode=args.mode,
            jobs=jobs,
            judge_contract_id=contract["contract_id"],
            model_id=contract["config"]["model_id"],
            model_revision=contract["config"]["model_revision"],
            new_cloud_compute=False,
            repeats_reader_inference=False,
        )
        if args.mode == "preview":
            print("EVALUATION_PLAN_PREVIEWED_NO_COMPUTE_CREATED")
            return 0
        session = boto3.Session(region_name=os.environ.get("AWS_REGION", "us-west-2"))
        client_config = Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
        )
        sagemaker = session.client("sagemaker", config=client_config)
        s3 = session.client("s3", config=client_config)
        infer = GemmaDecision(
            contract["config"], logger, root / "artifacts/semantic_judge/model_cache"
        )
        for index, job in enumerate(jobs, start=1):
            logger.emit(
                "evaluation.job.started", job_name=job, job_number=index, job_count=len(jobs)
            )
            with logger.stage("source.verify", heartbeat_seconds=15):
                run = load_saved_run(root, sagemaker, s3, job)
            logger.emit("evaluation.source.verified", job_name=job, question_count=len(run.records))
            save_public_evaluation(root, job, "supporting_metrics", supporting_metrics(run))
            if args.mode == "semantic":
                objects = ImmutableS3Objects(
                    s3,
                    run.bucket,
                    f"experiments/oracle-reader/evaluation/{run.source['protocol_lock_id']}/{contract['contract_id']}",
                )
                judge = DurableSemanticJudge(contract, objects, infer, logger)
                result = evaluate_semantic(run, judge)
                save_public_evaluation(root, job, "semantic_summary", result)
            logger.emit(
                "evaluation.job.completed", job_name=job, completed_jobs=index, total_jobs=len(jobs)
            )
        write_report(root)
    print("SAVED_PREDICTIONS_EVALUATION_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
