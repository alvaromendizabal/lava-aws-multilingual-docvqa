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
#   kernelspec:
#     display_name: Python (lava)
#     language: python
#     name: lava
# ---
# %% [markdown]
# # 00 — Reproducibility and protocol
#
# This notebook exposes the public, reproducible experiment contract without requiring private documents or paid cloud resources.
# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

ROOT = Path.cwd()
config = yaml.safe_load((ROOT / "configs/oracle_reader_benchmark.yaml").read_text())
protocol = json.loads((ROOT / "configs/evaluation_protocol.lock.json").read_text())
model_lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text())
assert config["protocol_lock_id"] == protocol["protocol_lock_id"]
# %% [markdown]
# ## Frozen model and hardware registry
# %%
models = pd.DataFrame(model_lock["resolved_models"])
columns = ["model_key", "model_id", "revision", "parameters_billion", "instance_type", "input_mode"]
display(models[columns].sort_values(["parameters_billion", "model_key"]).reset_index(drop=True))
# %% [markdown]
# ## Canonical no-cost preflight
#
# ```bash
# make quality
# make preflight MODEL=qwen35_9b_fused_direct
# make preview MODEL=qwen35_9b_fused_direct
# ```
#
# The 9B configuration resolves to the already-verified LAVA `ml.g6e.2xlarge` path. The 27B contract remains on `ml.g7e.12xlarge` because its frozen minimum-memory requirement is larger.
