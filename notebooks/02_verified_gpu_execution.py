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
# # 02 — Verified GPU execution
#
# SageMaker `Completed` is not treated as sufficient evidence. Public results are accepted only after the fail-closed artifact verifier succeeds.
# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
run_root = ROOT / "reports/oracle_reader/runs"
rows = []
if run_root.exists():
    for manifest_path in sorted(run_root.glob("*/sync_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        gate = manifest.get("artifact_gate", {})
        rows.append(
            {
                "job_name": manifest.get("job_name"),
                "model_key": manifest.get("model_key"),
                "instance_type": manifest.get("instance_type"),
                "training_seconds": manifest.get("training_time_seconds"),
                "billable_seconds": manifest.get("billable_time_seconds"),
                "raw_response_count": gate.get("raw_response_count"),
                "schema_valid_rate": gate.get("schema_valid_rate"),
                "parser_error_counts": gate.get("parser_error_counts"),
            }
        )

results = pd.DataFrame(rows)
display(
    results
    if not results.empty
    else pd.DataFrame({"status": ["Sync verified runs with `make sync JOB=...`."]})
)
# %% [markdown]
# ## Canonical operational commands
#
# ```bash
# make monitor JOB=<job-name>
# make verify JOB=<job-name>
# make sync JOB=<job-name>
# ```
#
# These commands are reconnectable and do not rely on notebook-kernel lifetime.
