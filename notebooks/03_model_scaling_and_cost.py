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
# The 16 questions from five documents are all supplied training labels
# (15 Japanese, one Vietnamese); this is not an arbitrary sample-size limit.
# It cannot establish language-general performance. Exact scores are diagnostics;
# the pinned Gemma semantic evaluation is also complete. Fold manifests describe a protocol,
# not completed nested training or tuning. Document-paired comparisons appear only
# after two compatible full pilots have passed the artifact gate.
#
# On the local LAVA overall metric, 9B gains 13.20 percentage points per question
# and 10.02 points per document. Three documents improve, one ties, and one
# regresses. The exploratory document-bootstrap interval spans -4.42 to +25.15
# points; the paired sign-flip p-value is 0.375. Five documents cannot establish
# broad superiority. Evidence F1 falls 3.13 points while semantic answer credit
# rises 29.52 points. The dashboard shows these metrics separately.
#
# The older normalized-exact diagnostic gains 7.65 points per question and loses
# 4.12 points per document; that comparison remains explicitly labeled and separate.
# GPU hardware also differs, so
# generation latency is an observed system comparison rather than isolated scaling.
# Both completed experiments should be reused; no additional inference is needed
# to inspect their results. The 9B run persisted and verified all 16 S3 checkpoints.
#
# Local semantic VQA is 50.625% for 4B and 80.149% for 9B; combined LAVA scores
# are 73.824% and 87.024%. These are local formula-based scores, not organizer-server
# results. All 28 public judge controls passed, and a second run reused every
# decision without model loading. Semantic charts use the current contract; the
# paired normalized-exact comparison remains separately labeled.
#
# `make report` writes the same standalone HTML and aggregate JSON for sharing.
#
# ## From oracle evaluation to a submission
#
# The 9B score deficit is concentrated in string and unordered-list questions.
# The report decomposes the deficit by answer format without claiming an error cause.
# Next, evaluate full-document evidence retrieval before reusing the reader with
# retrieved pages. The existing oracle answers cannot become test predictions.
#
# The test set has 624 questions from 200 documents (587 Japanese, 37 Vietnamese).
# Its labels are unavailable and must not be used for tuning. Broader architecture
# development needs a separate, appropriate public labeled dataset and a frozen
# evaluation plan. The organizer also requires inference within two hours on one
# A100 40GB GPU. Current per-question generation timings do not verify that limit.
#
# `make submission-preview` explains the required inputs without cloud access.
# `make submission-check` verifies the pinned test and sample-submission files in
# S3 and safely resumes from verified local copies. The exporter requires complete
# test predictions, PDF page counts and checksummed non-oracle provenance.
# See `docs/submission.md` for the schema and exact command.
#
# The normal competition closed on May 31, 2026. On September 6 the public Kaggle
# page shows a Late Submission button, disabled while signed out. Authenticated
# eligibility is unverified. No submission has been uploaded by this workflow.
