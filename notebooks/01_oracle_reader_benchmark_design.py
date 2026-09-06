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
# # 01 — Oracle-reader benchmark design
#
# The oracle-evidence benchmark holds evidence selection perfect so reader capability can be measured independently from retrieval quality.
# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
lock = json.loads((ROOT / "configs/oracle_reader_models.lock.json").read_text())
models = pd.DataFrame(lock["resolved_models"])
# %% [markdown]
# ## Controlled reader ladder
# %%
display(
    models[
        [
            "model_key",
            "parameters_billion",
            "input_mode",
            "instance_type",
            "dtype",
            "attention_implementation",
        ]
    ].sort_values(["parameters_billion", "model_key"])
)
# %% [markdown]
# The experimental sequence is deliberately controlled:
#
# 1. deterministic fused 4B baseline;
# 2. image-only and text-only modality controls;
# 3. bounded thinking-mode ablation;
# 4. 9B capacity challenger on the verified G6e path;
# 5. 27B high-memory challenger;
# 6. multilingual slices, error taxonomy, latency/throughput/VRAM, and cost-quality Pareto analysis;
# 7. retrieval and reranking only after reader selection.
