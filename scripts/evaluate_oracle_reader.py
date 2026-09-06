"""Score saved complete pilots; reader inference and new cloud compute are never launched."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

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
    parser.add_argument("--mode", choices=("preview", "diagnostics", "semantic"), default="preview")
    args = parser.parse_args()
    root = find_repo_root(Path(__file__).resolve())
    load_dotenv(root / ".env", override=False)
    logger = RuntimeEventLogger("oracle_reader.evaluation")
    with logger.stage("evaluation", heartbeat_seconds=15):
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
        for job in jobs:
            with logger.stage("source.verify", heartbeat_seconds=15):
                run = load_saved_run(root, sagemaker, s3, job)
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
        write_report(root)
    print("SAVED_PREDICTIONS_EVALUATION_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
