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
# # 01 — Oracle-evidence multimodal reader benchmark
#
# This notebook compares reader architectures while page retrieval is held perfect.
# Public displays contain sanitized aggregate metadata only. Private questions,
# answers, document IDs, page images, extracted text, bucket names, and S3 version
# identifiers remain outside the committed notebook.

# %%
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from IPython.display import display

from lava.notebook_support import find_repo_root, load_project_environment, public_metadata
from lava.observability import EventLogger
from lava.readers.sagemaker import build_job_plan

ROOT = find_repo_root()
os.chdir(ROOT)
environment = load_project_environment(ROOT)
RUN_ID = f"notebook-01-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
LOGGER = EventLogger.to_stdout(
    run_id=RUN_ID,
    component="notebook.01_oracle_reader",
    jsonl_path=ROOT / "artifacts" / "notebook_runs" / f"{RUN_ID}.jsonl",
)

# %% [markdown]
# ## Frozen private assets and immutable model revisions

# %%
with LOGGER.stage("load_locks", heartbeat_seconds=15.0):
    asset_summary = json.loads(Path("reports/oracle_reader/oracle_assets_summary.json").read_text())
    model_lock = json.loads(Path("configs/oracle_reader_models.lock.json").read_text())
    public_asset_summary = public_metadata(asset_summary)
    LOGGER.emit(
        "oracle_assets.loaded",
        question_count=asset_summary["question_count"],
        document_count=asset_summary["document_count"],
        unique_evidence_page_count=asset_summary["unique_evidence_page_count"],
        protocol_lock_id=asset_summary["protocol_lock_id"],
    )
    print(json.dumps(public_asset_summary, indent=2, sort_keys=True))

# %%
models = pd.DataFrame(model_lock["resolved_models"])
display(
    models[
        [
            "model_key",
            "model_id",
            "revision",
            "parameters_billion",
            "instance_type",
            "input_mode",
            "attention_implementation",
        ]
    ]
)

# %% [markdown]
# ## Build the first one-question smoke plan
#
# This cell is pure planning. It does not create a trainer, endpoint, or AWS job.

# %%
with LOGGER.stage("build_smoke_plan", heartbeat_seconds=15.0):
    plan = build_job_plan(
        repo_root=ROOT,
        config_path=Path("configs/oracle_reader_benchmark.yaml"),
        model_lock_path=Path("configs/oracle_reader_models.lock.json"),
        model_key="qwen35_4b_fused_direct",
        bucket=environment["S3_BUCKET"],
        limit=1,
    )
    public_plan = public_metadata(plan.model_dump(mode="json"))
    LOGGER.emit("smoke_plan.built", plan=public_plan)
    print(json.dumps(public_plan, indent=2, sort_keys=True))

# %% [markdown]
# ## Paid execution gate
#
# Preview the fully validated plan first:
#
# ```bash
# uv run python scripts/run_oracle_reader_smoke.py
# ```
#
# Only after the preview and quota checks pass, submit exactly one question:
#
# ```bash
# uv run python scripts/run_oracle_reader_smoke.py \
#   --submit \
#   --wait \
#   --acknowledge-charges YES
# ```
#
# The wrapper emits UTC timestamps, status changes, 30-second heartbeats, total
# elapsed time, conservative cost bounds, and a reconnectable local state file.

# %% [markdown]
# ## Controlled reader ladder
#
# 1. Fused page image plus native text, deterministic direct decoding.
# 2. Image-only and text-only controls at the same immutable model revision.
# 3. Bounded thinking-mode reader.
# 4. Larger 9B challenger after memory and latency profiling.
# 5. Render-resolution and answer-blind region-crop ablations.
# 6. Pinned semantic judge and repeated-run stability.
#
# Reader selection precedes retrieval development so page-selection failures cannot
# hide reader quality.
