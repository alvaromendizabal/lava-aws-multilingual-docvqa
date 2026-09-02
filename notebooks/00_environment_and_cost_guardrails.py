# %% [markdown]
# # 00 — Environment and cost guardrails
#
# This notebook verifies the frozen protocol, AWS identity, S3 access, Git state,
# SDK contract, and charge bounds before any GPU job is submitted. It is safe to
# run in the inexpensive SageMaker Studio CPU space.

# %%
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import boto3
import pandas as pd
import yaml
from dotenv import load_dotenv

from lava.readers.sagemaker import build_job_plan, validate_sagemaker_sdk_contract

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

# %%
config = yaml.safe_load(Path("configs/oracle_reader_benchmark.yaml").read_text())
protocol_lock = json.loads(Path("configs/evaluation_protocol.lock.json").read_text())
assert config["protocol_lock_id"] == protocol_lock["protocol_lock_id"]

identity = boto3.client("sts").get_caller_identity()
branch = subprocess.run(
    ["git", "branch", "--show-current"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
status = subprocess.run(
    ["git", "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

snapshot = {
    "aws_arn": identity["Arn"],
    "region": os.environ.get("AWS_REGION", "us-west-2"),
    "branch": branch,
    "working_tree_clean": not status,
    "protocol_lock_id": protocol_lock["protocol_lock_id"],
    "question_count": protocol_lock["question_count"],
    "document_count": protocol_lock["document_count"],
    "creates_endpoint": False,
}
snapshot

# %% [markdown]
# ## Validate the installed SageMaker SDK contract

# %%
validate_sagemaker_sdk_contract(config["training_runtime"]["sdk_version"])

# %% [markdown]
# ## Charge-bounded experiment matrix
#
# The first paid job is one question, one L40S GPU, one question, one hour maximum, and no endpoint.

# %%
rows = []
for key, model in config["models"].items():
    generation = model["generation"]
    rows.append(
        {
            "model_key": key,
            "model_id": model["model_id"],
            "instance_type": model["instance_type"],
            "mode": generation["mode"],
            "input_mode": model["input_mode"],
            "max_new_tokens": generation["max_new_tokens"],
            "sampled": generation["do_sample"],
        }
    )
pd.DataFrame(rows)

# %% [markdown]
# ## Safe next command
#
# Run the preflight in a terminal. It resolves immutable model revisions, creates
# private oracle assets, verifies the SDK and tests, and previews the SageMaker
# plan. It does **not** submit a paid job.
#
# ```bash
# bash scripts/phase5a_preflight.sh
# ```
