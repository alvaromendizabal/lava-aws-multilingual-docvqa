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
# # 02 — Follow a GPU experiment from launch to evidence
#
# This notebook is the operational companion to Notebook 03. Run its cells to
# inspect saved, verified work. It does not launch a GPU or repeat model inference.
#
# ## 1. See which stage each model has reached
#
# A **smoke** checks one question. A **full pilot** covers all 16 supplied labels.
# **Semantic scoring** then evaluates those saved answers with the pinned judge.
# These are distinct milestones, so "Completed" in SageMaker alone is insufficient.
# %%
import json
from pathlib import Path

from IPython.display import HTML, display

from lava.evaluation.reporting import load_report
from lava.evaluation.walkthrough import TABLE_STYLE, comparison_tables, render_table, training_rates
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = find_repo_root(Path.cwd())
logger = RuntimeEventLogger("notebook.verified_execution")
with logger.stage("01_verified_coverage", heartbeat_seconds=15):
    report = load_report(ROOT)
    pricing = json.loads((ROOT / "reports/aws/training_prices.json").read_text())
    tables = comparison_tables(report, training_rates(pricing))
    display(
        HTML(TABLE_STYLE + render_table(tables["coverage"], caption="Verified experiment stages"))
    )
# %% [markdown]
# ## 2. Understand the cloud lifecycle
#
# | Stage | What is happening | What to watch |
# |---|---|---|
# | Pending | AWS is finding capacity | Queue elapsed time and server pending limit |
# | Downloading | The instance downloads its container and inputs | SageMaker phase changes |
# | Training | This job runs inference: weight loading, then questions | CloudWatch stage heartbeats and question counts |
# | Uploading | AWS persists the final model-output artifacts | Upload completion |
# | Completed | The process exited successfully | Artifact verification must still pass |
#
# Here, SageMaker calls the service a *training job*, but our workload evaluates
# frozen models; it does not fine-tune weights. Once provisioned, setup time can
# be billable even before the first answer. Pending capacity time is separate.
# Studio, S3 and CloudWatch have their own charges.
#
# The current job contract allows 24 hours pending and one hour runtime. Those
# limits live in AWS. Closing a browser or losing a notebook kernel does not stop
# the managed job. A terminal monitor may disconnect; reconnect to the same job
# instead of submitting another one.
# %%
with logger.stage("02_saved_run_history", heartbeat_seconds=15):
    rows = [
        {
            "Reader": run["label"],
            "Coverage": "Full pilot" if run["complete"] else "Smoke",
            "Questions": run["summary"]["record_count"],
            "Instance": run["instance_type"],
            "Billable time (s)": run["billable_seconds"],
            "Job": run["job_name"],
        }
        for run in report["runs"]
    ]
    display(HTML(render_table(rows, caption="Saved history · live status comes from the monitor")))
# %% [markdown]
# ## 3. Use one canonical operational interface
#
# Run these commands in the project terminal. Replace `<job-name>` with the
# exact name printed at submission or shown above.
#
# ```bash
# make benchmark-preflight MODEL=qwen38_27b_nf4_g5_fused_direct
# make benchmark-preview MODEL=qwen38_27b_nf4_g5_fused_direct
# make monitor JOB=<job-name>
# make verify JOB=<job-name>
# make sync JOB=<job-name>
# make evaluate JOB=<job-name>
# make report
# ```
#
# Preflight and preview create no GPU. `monitor` reconnects, `verify` checks
# artifacts, `sync` imports verified public summaries, and `evaluate` scores saved
# answers on the existing CPU workspace. None of these commands repeats reader
# inference. `docs/benchmark.md` documents the explicit paid submission command.
#
# ## 4. Recover without discarding correct work
#
# Each completed answer has a durable S3 checkpoint with its identity and
# checksums. If a benchmark **fails or is stopped**, preview a resume plan:
#
# ```bash
# make benchmark-resume-preview MODEL=qwen38_27b_nf4_g5_fused_direct JOB=<job-name>
# ```
#
# It verifies which answers are reusable. A paid resume creates a new bounded
# attempt using those answers; it does not silently relaunch a running/completed
# job. A crash before an answer's checkpoint is written may require that one
# question again. Immutable judge decisions are similarly reused when scoring.
#
# Code and public aggregate results belong in GitHub. Raw answers, checkpoints
# and executed notebook outputs belong in durable S3 artifacts. GPU scratch disks
# are working space, not the only copy of valuable results. CloudWatch carries
# runtime heartbeats; persisted notebook outputs show completed analysis steps.
# %%
logger.emit(
    "operations_walkthrough.completed",
    verified_runs=len(report["runs"]),
    cloud_resources_created=0,
)
