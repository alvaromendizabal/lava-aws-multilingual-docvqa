"""Verify the targeted recovery data path under its real execution role, without inference."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from lava.evaluation.semantic import digest
from lava.evaluation.submission_store import cached_source
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import store_for


def verify_caller(identity: dict, role_arn: str) -> None:
    """An administrator's successful read cannot stand in for the job role's access."""
    account, role_name = role_arn.split(":")[4], role_arn.rsplit("/", 1)[1]
    expected = f"arn:aws:sts::{account}:assumed-role/{role_name}/"
    if identity.get("Account") != account or not identity.get("Arn", "").startswith(expected):
        raise ValueError("Access probe must execute as the exact SageMaker role")


def verify_asset(s3: Any, bucket: str, asset: dict, kind: str) -> None:
    request = {
        "Bucket": bucket,
        "Key": asset[f"{kind}_s3_uri"].split(f"s3://{bucket}/", 1)[1],
    }
    version = asset.get(f"{kind}_version_id")
    if version is not None:
        request["VersionId"] = version
    response = s3.get_object(**request)
    with response["Body"] as stream:
        actual = stream.read()
    if digest(actual) != asset[f"{kind}_sha256"]:
        raise ValueError("Rendered asset checksum mismatch")
    observed = response.get("VersionId")
    if not observed or (version is not None and observed != version):
        raise ValueError("Rendered asset version mismatch")
    # Ordinary page assets have content hashes but no pinned version in the schema.
    # Also prove that the job can retrieve the exact version just read.
    if version is None:
        verify_asset(s3, bucket, {**asset, f"{kind}_version_id": observed}, kind)


def run(root: Path, s3: Any, sts: Any, env: dict) -> dict:
    # Load the deployed pipeline at runtime; its source is bound by contract_for.
    # This probe must not require edits to the immutable inference implementation.
    targeted = import_module("pipelines.submission.targeted")
    identity = sts.get_caller_identity()
    verify_caller(identity, env["LAVA_EXPECTED_ROLE_ARN"])
    bucket = env["LAVA_BUCKET"]
    plan = targeted.read_pinned(s3, bucket, json.loads(env["LAVA_TARGETED_PLAN"]))
    contract = targeted.contract_for(root, plan)
    if contract["contract_id"] != env["LAVA_TARGETED_CONTRACT"]:
        raise ValueError("Access probe source differs from targeted contract")
    logger = RuntimeEventLogger("submission.access")
    with logger.stage("parent.verify", heartbeat_seconds=15):
        _, _, requests, preserved = targeted.verify_parent(root, s3, bucket, plan)
    renderer = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    originals = {r.question_id: r for r in requests}
    checked_assets, checked_documents = 0, set()
    with logger.stage("routed.assets.verify", heartbeat_seconds=15):
        for qid, routes in sorted(plan["routes"].items()):
            original = originals[qid]
            page = original.pages[0]
            source = {
                "key": page.source_pdf_s3_uri.split(f"s3://{bucket}/", 1)[1],
                "sha256": page.source_pdf_sha256,
                "version_id": page.source_pdf_version_id,
            }
            payload = cached_source(
                s3,
                bucket,
                root / "artifacts/submission/inputs" / f"{original.document_id}.pdf",
                source,
                logger,
            )
            checked_documents.add(original.document_id)
            for route in routes:
                for number in route["pages"]:
                    asset = targeted.prepare_asset(
                        s3,
                        bucket,
                        payload,
                        source,
                        original,
                        number,
                        route["crops"].get(str(number)),
                        render,
                        contract["contract_id"],
                    )
                    for kind in ("image", "text"):
                        verify_asset(s3, bucket, asset, kind)
                    checked_assets += 1
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "contract_id": contract["contract_id"],
        "execution_role_arn": env["LAVA_EXPECTED_ROLE_ARN"],
        "caller_arn": identity["Arn"],
        "source_commit": env["LAVA_GIT_COMMIT_SHA"],
        "probe_sha256": digest(Path(__file__).read_bytes()),
        "preserved_count": preserved,
        "routed_question_count": len(plan["routes"]),
        "source_blocker_count": len(plan["blocked"]),
        "verified_document_count": len(checked_documents),
        "verified_page_asset_count": checked_assets,
        "model_calls": 0,
        "csv_exported": False,
    }
    store = store_for(s3, bucket, contract["contract_id"])
    name = "diagnostics/access-preflight.json"
    store.write(name, report)
    if store.read(name) != report:
        raise ValueError("Access receipt did not survive durable read-back")
    return report


def main() -> None:
    import boto3

    region = os.environ["AWS_DEFAULT_REGION"]
    env = dict(os.environ)
    if "LAVA_TARGETED_PLAN" not in env:
        # Processing environment values are capped at 256 characters.
        env["LAVA_TARGETED_PLAN"] = json.dumps(
            {
                "key": env["LAVA_PLAN_KEY"],
                "version_id": env["LAVA_PLAN_VERSION"],
                "sha256": env["LAVA_PLAN_SHA256"],
            }
        )
    report = run(
        Path("/opt/ml/code"),
        boto3.client("s3", region_name=region),
        boto3.client("sts", region_name=region),
        env,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
