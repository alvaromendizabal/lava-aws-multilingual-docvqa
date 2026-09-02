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
# Public outputs contain aggregate metadata only. Private questions, answers,
# document IDs, page images, and extracted text remain in S3.

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from lava.readers.sagemaker import build_job_plan

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

# %%
asset_summary = json.loads(Path("reports/oracle_reader/oracle_assets_summary.json").read_text())
model_lock = json.loads(Path("configs/oracle_reader_models.lock.json").read_text())
print(json.dumps(asset_summary, indent=2, sort_keys=True))

# %%
models = pd.DataFrame(model_lock["resolved_models"])
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

# %% [markdown]
# ## Build the first smoke plan
#
# This cell is pure planning. It does not construct a trainer or create an AWS job.

# %%
plan = build_job_plan(
    repo_root=ROOT,
    config_path=Path("configs/oracle_reader_benchmark.yaml"),
    model_lock_path=Path("configs/oracle_reader_models.lock.json"),
    model_key="qwen35_4b_fused_direct",
    bucket=os.environ["S3_BUCKET"],
    limit=1,
)
print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))

# %% [markdown]
# ## Paid execution gate
#
# Commit the code and immutable locks before running this terminal command:
#
# ```bash
# uv run python scripts/launch_oracle_reader.py \
#   --model-key qwen35_4b_fused_direct \
#   --limit 1 \
#   --submit \
#   --wait \
#   --acknowledge-charges YES
# ```
#
# Then synchronize the checksum-verified public result:
#
# ```bash
# uv run python scripts/sync_oracle_reader_results.py
# ```

# %% [markdown]
# ## Controlled reader ladder
#
# 1. Fused page image plus native text, direct deterministic mode.
# 2. Image-only and text-only controls at the same immutable model revision.
# 3. Thinking-mode reader with bounded seeded sampling.
# 4. Larger 9B challenger after memory and runtime measurements.
# 5. Render-resolution and answer-blind region-crop ablations.
# 6. Pinned semantic judge and repeated-run stability.
#
# Reader selection precedes retrieval development so page-selection failures cannot
# hide reader quality.
