# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (lava)
#     language: python
#     name: lava
# ---

# %% [markdown]
# # 00 — Environment, protocol, and cost guardrails
#
# This notebook verifies AWS identity, private S3 access, the immutable evaluation
# protocol, Git provenance, the SageMaker SDK contract, and charge bounds before
# any GPU job is submitted. It is safe to run in the inexpensive Studio CPU space.
#
# Public notebook outputs are stripped before commit. Raw questions, answers,
# document identifiers, page images, and bucket names remain private.

# %%
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pandas as pd
import yaml
from IPython.display import display

from lava.notebook_support import (
    find_repo_root,
    git_snapshot,
    load_project_environment,
    public_metadata,
)
from lava.observability import EventLogger, estimate_maximum_cost
from lava.readers.sagemaker import validate_sagemaker_sdk_contract

ROOT = find_repo_root()
os.chdir(ROOT)
environment = load_project_environment(ROOT)
RUN_ID = f"notebook-00-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
LOGGER = EventLogger.to_stdout(
    run_id=RUN_ID,
    component="notebook.00_environment",
    jsonl_path=ROOT / "artifacts" / "notebook_runs" / f"{RUN_ID}.jsonl",
)

# %% [markdown]
# ## Verify immutable protocol and AWS access

# %%
with LOGGER.stage("environment_validation", heartbeat_seconds=15.0):
    config = yaml.safe_load(Path("configs/oracle_reader_benchmark.yaml").read_text())
    protocol_lock = json.loads(Path("configs/evaluation_protocol.lock.json").read_text())
    assert config["protocol_lock_id"] == protocol_lock["protocol_lock_id"]

    session = boto3.session.Session(region_name=environment["AWS_REGION"])
    identity = session.client("sts").get_caller_identity()
    session.client("s3").head_bucket(Bucket=environment["S3_BUCKET"])

    counts = {
        "question_count": protocol_lock.get(
            "expected_question_count",
            protocol_lock.get("question_count"),
        ),
        "document_count": protocol_lock.get(
            "expected_document_count",
            protocol_lock.get("document_count"),
        ),
    }
    snapshot = {
        **git_snapshot(ROOT),
        "aws_arn": identity["Arn"],
        "region": environment["AWS_REGION"],
        "protocol_lock_id": protocol_lock["protocol_lock_id"],
        **counts,
        "s3_access_verified": True,
        "creates_endpoint": False,
    }
    LOGGER.emit("environment.verified", **snapshot)
    print(json.dumps(public_metadata(snapshot), indent=2, sort_keys=True))

# %% [markdown]
# ## Validate the installed SageMaker SDK contract

# %%
with LOGGER.stage("sagemaker_sdk_validation", heartbeat_seconds=15.0):
    validate_sagemaker_sdk_contract(config["training_runtime"]["sdk_version"])
    LOGGER.emit(
        "sagemaker_sdk.verified",
        expected_version=config["training_runtime"]["sdk_version"],
    )

# %% [markdown]
# ## Charge-bounded experiment matrix
#
# The first paid run is constrained to one question, one instance, one hour,
# no persistent endpoint, and an operator-supplied conservative cost ceiling.

# %%
rows = []
for key, model in config["models"].items():
    generation = model["generation"]
    estimate = estimate_maximum_cost(
        hourly_usd_ceiling=10.0,
        max_runtime_seconds=config["training_runtime"]["max_runtime_seconds"],
        instance_count=1,
        contingency_factor=1.25,
    )
    rows.append(
        {
            "model_key": key,
            "model_id": model["model_id"],
            "instance_type": model["instance_type"],
            "mode": generation["mode"],
            "input_mode": model["input_mode"],
            "max_new_tokens": generation["max_new_tokens"],
            "sampled": generation["do_sample"],
            "operator_cost_ceiling_usd": estimate.estimated_maximum_usd,
        }
    )
display(pd.DataFrame(rows))

# %% [markdown]
# ## Safe next command
#
# Run the no-cost observable preflight in a terminal:
#
# ```bash
# bash scripts/phase5b_preflight.sh
# ```
#
# It runs all tests, validates notebooks, checks the immutable model and asset
# locks, inspects the SageMaker quota, verifies that no LAVA job is active, and
# previews the first one-question plan. It does **not** submit a paid resource.
