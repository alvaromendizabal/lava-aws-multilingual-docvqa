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
# # 02 — Verified GPU execution
#
# SageMaker `Completed` is not treated as sufficient evidence. Public results are accepted only after the fail-closed artifact verifier succeeds.
# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd
from IPython.display import display

from lava.evaluation.reporting import load_report
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = find_repo_root(Path.cwd())
logger = RuntimeEventLogger("notebook.verified_execution")
with logger.stage("coverage_and_performance", heartbeat_seconds=15):
    report = load_report(ROOT)
    rows = []
    for run in report["current_models"]:
        summary = run["summary"]
        semantic = run["semantic_summary"]
        rows.append(
            {
                "reader": run["label"],
                "coverage": f"{summary['record_count']} / {report['expected_questions']}",
                "status": "Full pilot verified" if run["complete"] else "Smoke only",
                "answer_diagnostic": summary["normalized_exact_answer_micro"]
                if run["complete"]
                else None,
                "evidence_f1": summary["self_grounding_f1_micro"] if run["complete"] else None,
                "local_lava_score": semantic["metrics"]["question_micro"]["overall"]
                if semantic and semantic["contract_current"]
                else "Not evaluated",
                "valid_output": summary["schema_valid_rate"],
                "billable_seconds": run["billable_seconds"],
                "instance": run["instance_type"],
            }
        )
    display(pd.DataFrame(rows))
# %% [markdown]
# Full pilots take precedence over historical smoke tests. 4B and 9B each have
# 16-question results; 27B currently has a one-question smoke. The official formula
# averages semantic VQA and evidence-page F1. A missing Gemma score is not zero.
# These readers receive oracle evidence, so their citation F1 is not a retrieval score.
# Notebook 03 contains the complete metrics, document comparisons, and run history.
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
