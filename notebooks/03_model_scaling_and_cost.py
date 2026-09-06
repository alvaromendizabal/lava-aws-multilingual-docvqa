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
# # 03 — Reader evaluation and systems cost
#
# This notebook regenerates an offline report from checksum-verified public artifacts.
# Smoke tests and complete 16-question pilots have distinct coverage labels.
# Every complete pilot retains parsing failures and abstentions in its denominator.
# %%
from pathlib import Path

from IPython.display import HTML, display

from lava.evaluation.reporting import load_report, render_report
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = find_repo_root(Path.cwd())
logger = RuntimeEventLogger("notebook.reader_evaluation")
with logger.stage("report", heartbeat_seconds=15):
    report = load_report(ROOT)
    display(HTML(render_report(report)))
# %% [markdown]
# ## Interpretation
#
# The pilot has 16 questions from five documents (15 Japanese, one Vietnamese).
# It cannot establish language-general performance. Exact scores are diagnostics;
# semantic judging remains a separate milestone. Fold manifests describe a protocol,
# not completed nested training or tuning. Document-paired comparisons appear only
# after two compatible full pilots have passed the artifact gate.
#
# The verified 4B and 9B pilots show why both weighting schemes matter: 9B gains
# 7.65 percentage points per question but loses 4.12 points per document. Two
# documents improve, two tie, and one regresses. The dashboard exposes the signed
# document changes and the exploratory interval. GPU hardware also differs, so
# generation latency is an observed system comparison rather than isolated scaling.
# Both completed experiments should be reused; no additional inference is needed
# to inspect their results. The 9B run persisted and verified all 16 S3 checkpoints.
#
# `make report` writes the same standalone HTML and aggregate JSON for sharing.
