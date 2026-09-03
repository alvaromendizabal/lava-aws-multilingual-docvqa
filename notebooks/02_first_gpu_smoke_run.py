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
# # 02 — First observable GPU smoke run
#
# This notebook reviews the current immutable plan and any existing run state.
# The paid job is launched from a terminal through a fail-closed wrapper so an
# interrupted notebook kernel cannot hide or duplicate cloud execution.

# %%
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from IPython.display import display

from lava.notebook_support import find_repo_root, load_project_environment, public_metadata
from lava.observability import EventLogger, SageMakerTrainingMonitor, latest_state_path, read_state
from lava.readers.sagemaker import build_job_plan

ROOT = find_repo_root()
os.chdir(ROOT)
environment = load_project_environment(ROOT)
RUN_ID = f"notebook-02-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
LOGGER = EventLogger.to_stdout(
    run_id=RUN_ID,
    component="notebook.02_gpu_smoke",
    jsonl_path=ROOT / "artifacts" / "notebook_runs" / f"{RUN_ID}.jsonl",
)

# %% [markdown]
# ## Review the immutable one-question plan

# %%
with LOGGER.stage("review_plan", heartbeat_seconds=15.0):
    plan = build_job_plan(
        repo_root=ROOT,
        config_path=Path("configs/oracle_reader_benchmark.yaml"),
        model_lock_path=Path("configs/oracle_reader_models.lock.json"),
        model_key="qwen35_4b_fused_direct",
        bucket=environment["S3_BUCKET"],
        limit=1,
    )
    display(public_metadata(plan.model_dump(mode="json")))

# %% [markdown]
# ## Inspect reconnectable state, when a job has already been submitted

# %%
state_path = latest_state_path(ROOT)
if state_path.is_file():
    state = read_state(state_path)
    display(public_metadata(state.as_dict()))
    if state.job_name:
        client = boto3.session.Session(region_name=environment["AWS_REGION"]).client("sagemaker")
        monitor = SageMakerTrainingMonitor(
            sagemaker_client=client,
            logger=LOGGER,
        )
        display(public_metadata(monitor.describe(state.job_name).as_dict()))
else:
    LOGGER.emit("run_state.absent", message="No paid smoke job has been submitted yet.")
    print("No paid smoke job has been submitted yet.")

# %% [markdown]
# ## Terminal commands
#
# Preview without creating a paid resource:
#
# ```bash
# uv run python scripts/run_oracle_reader_smoke.py
# ```
#
# Submit the single-question job only after preflight and quota approval:
#
# ```bash
# uv run python scripts/run_oracle_reader_smoke.py \
#   --submit \
#   --wait \
#   --acknowledge-charges YES
# ```
#
# Reconnect to a job after a terminal or browser interruption:
#
# ```bash
# uv run python scripts/monitor_oracle_reader_job.py
# ```
