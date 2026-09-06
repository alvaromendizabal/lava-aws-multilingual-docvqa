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
# # 03 — Model scaling and systems dashboard
#
# This notebook reads only sanitized committed run manifests. It becomes the public dashboard for model-size scaling, reliability, runtime, and cost-facing comparisons as 4B, 9B, and 27B benchmark runs accumulate.
# %%
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
run_root = ROOT / "reports/oracle_reader/runs"
rows = []
if run_root.exists():
    for manifest_path in sorted(run_root.glob("*/sync_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        gate = manifest.get("artifact_gate", {})
        model_key = str(manifest.get("model_key") or "")
        match = re.search(r"_(\d+)b_", model_key)
        rows.append(
            {
                "model_key": model_key,
                "parameters_billion": float(match.group(1)) if match else None,
                "instance_type": manifest.get("instance_type"),
                "billable_seconds": manifest.get("billable_time_seconds"),
                "schema_valid_rate": gate.get("schema_valid_rate"),
                "raw_response_count": gate.get("raw_response_count"),
            }
        )
results = pd.DataFrame(rows)
display(results)
# %% [markdown]
# ## Billable runtime versus model size
# %%
if len(results) >= 2 and results["parameters_billion"].notna().all():
    plot_data = results.dropna(subset=["parameters_billion", "billable_seconds"]).sort_values(
        "parameters_billion"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_data["parameters_billion"], plot_data["billable_seconds"], marker="o")
    ax.set_xlabel("Model parameters (billions)")
    ax.set_ylabel("SageMaker billable seconds")
    ax.set_title("Reader scaling: model size vs. billable runtime")
    ax.grid(alpha=0.25)
    plt.show()
else:
    print("At least two synchronized runs are required for a scaling curve.")
# %% [markdown]
# Future benchmark cells add answer-quality metrics, multilingual slices, latency, throughput, peak VRAM, and cost-quality Pareto fronts. The notebook intentionally does not invent metrics that have not yet been produced by the frozen evaluator.
