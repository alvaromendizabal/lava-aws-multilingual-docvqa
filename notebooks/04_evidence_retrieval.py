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
# # 04 — Can we find the evidence in a complete PDF?
#
# Notebook 03 compared readers **after giving them the correct evidence pages**.
# A useful document-QA system must first find those pages. This notebook explains
# and inspects a completed retrieval experiment. It reads verified public results;
# running its cells does not download PDFs, launch GPUs, or repeat model inference.
#
# ## 1. Know which experiment is being measured
#
# We freeze a multilingual **BM25 text-search baseline** and compare it with taking
# pages in their original order. Every physical page is eligible. Rankings use only
# the question and PDF text. Gold answers and evidence pages enter only afterward,
# when the evaluator scores the saved rankings. No parameter was selected from these
# results, and no test questions or labels were used.
# %%
import json
from pathlib import Path

from IPython.display import HTML, display

from lava.evaluation.walkthrough import TABLE_STYLE, render_table
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.pipeline import load_public_report
from lava.retrieval.reporting import coverage_rows, document_rows, metric_rows, recall_chart

ROOT = find_repo_root(Path.cwd())
logger = RuntimeEventLogger("notebook.retrieval")
with logger.stage("01_load_verified_retrieval", heartbeat_seconds=15):
    summary = load_public_report(ROOT)
    display(
        HTML(
            TABLE_STYLE
            + render_table(
                [
                    {"Item": "Status", "Value": summary["status"]},
                    {"Item": "Labeled questions", "Value": summary["question_count"]},
                    {"Item": "PDFs", "Value": summary["document_count"]},
                    {
                        "Item": "Physical pages searched",
                        "Value": sum(row["page_count"] for row in summary["document_coverage"]),
                    },
                    {
                        "Item": "Reader answer score",
                        "Value": "Not evaluated with retrieved pages yet",
                    },
                ],
                caption="What has actually completed?",
            )
        )
    )

# %% [markdown]
# ## 2. Check extraction coverage before trusting any score
#
# PyMuPDF extracts the native text layer from **every page**, in physical PDF order.
# Textless pages stay in the index with no lexical signal; extraction failures stay
# visible. A page can contain native text and still need visual understanding for a
# chart or table. Native-text availability is not proof that all useful content was read.
#
# This pilot contains all 16 supplied training labels: 15 Japanese and one Vietnamese
# question. The five PDFs, not the 16 questions, are the more useful unit for checking
# consistency. The single Vietnamese example cannot estimate language-wide performance.
# %%
with logger.stage("02_check_page_coverage", heartbeat_seconds=15):
    display(HTML(render_table(coverage_rows(summary), caption="No PDF pages silently dropped")))
    print(f"Questions with zero lexical signal: {summary['zero_signal_questions']}")

# %% [markdown]
# ## 3. Understand what the retriever does
#
# Text is normalized with Unicode NFKC and case folding. We retain words plus
# character bigrams and trigrams, so Japanese matching does not depend on spaces.
# Vietnamese diacritics remain intact. BM25 rewards relevant term matches while
# accounting for page length; repeated occurrences have diminishing returns.
# Document frequencies come only from the PDF being searched. A page-number tie
# break makes the ranking deterministic. Empty/no-match queries fall back to that
# order and are counted as zero-signal cases.
#
# This is an interpretable baseline, not a trained visual retriever. It establishes
# the reference that a future multilingual embedding or visual method must improve.
# %%
with logger.stage("03_inspect_frozen_method", heartbeat_seconds=15):
    configuration = summary["implementation"]["config"]
    display(
        HTML(
            render_table(
                [
                    {"Setting": "BM25 k1", "Value": configuration["bm25"]["k1"]},
                    {"Setting": "BM25 b", "Value": configuration["bm25"]["b"]},
                    {
                        "Setting": "Page budgets",
                        "Value": ", ".join(map(str, configuration["budgets"])),
                    },
                    {"Setting": "OCR", "Value": "Not used in this baseline"},
                    {"Setting": "Hyperparameter tuning", "Value": "None"},
                ],
                caption="Settings chosen before looking at scores",
            )
        )
    )

# %% [markdown]
# ## 4. Measure both partial and complete evidence
#
# **Recall@k** is the fraction of gold evidence pages in the first k results.
# **All-evidence@k** succeeds only when every required page is present. A multi-page
# answer may fail even when recall looks high.
#
# MRR@k rewards the first relevant hit. MAP@k rewards relevant pages occurring early;
# this project's AP@k denominator is `min(number of relevant pages, k)`.
# nDCG@k compares ranked relevance with an ideal ordering. All budgets were fixed
# before this run. Report the entire curve rather than choosing k from the best-looking row.
# A PDF with fewer than k pages contributes all its physical pages.
# %%
with logger.stage("04_measure_retrieval", heartbeat_seconds=15):
    display(HTML(recall_chart(summary)))
    display(HTML(recall_chart(summary, "all_evidence_at_k")))
    display(HTML(render_table(metric_rows(summary), caption="Question-weighted metrics")))

# %% [markdown]
# ## 5. Check whether one PDF drives the result
#
# A question average gives more weight to PDFs with more questions. The second
# table gives each PDF equal weight. Large differences across PDFs are a reason
# to investigate failure modes and expand document-isolated evaluation.
# These fixed-model diagnostics are not cross-validation results or evidence of
# generalization to the 200 hidden-label test PDFs.
# %%
with logger.stage("05_compare_documents", heartbeat_seconds=15):
    display(
        HTML(render_table(document_rows(summary), caption="BM25 complete-evidence coverage by PDF"))
    )
    display(
        HTML(
            render_table(metric_rows(summary, "document_average"), caption="Equal-document metrics")
        )
    )

# %% [markdown]
# ## 6. Verify persistence and elapsed time
#
# The CPU runner verifies pinned source versions and hashes. Each completed PDF
# extraction and question ranking is stored in S3 and read back before progress is
# acknowledged. A fresh process restores these checkpoints. Conditional writes
# protect completed work from another writer; corrupted or incompatible outputs
# are rejected. Incomplete work is retried on the next run.
#
# The current CPU process uses Studio. If Studio stops, its process stops; completed
# checkpoints remain in S3 and the same command resumes them. Long GPU experiments
# use independent SageMaker jobs. Completed analysis notebooks have separate checksum
# manifests so viewing results does not require repeating either workload.
# %%
with logger.stage("06_verify_runtime_and_resume", heartbeat_seconds=15):
    validation = json.loads((ROOT / "reports/retrieval/validation.json").read_text())
    rows = [
        {
            "Run": row["run"],
            "Elapsed seconds": row["total_elapsed_seconds"],
            "PDFs computed": row["documents_computed"],
            "PDFs reused": row["documents_reused"],
            "Queries computed": row["queries_computed"],
            "Queries reused": row["queries_reused"],
        }
        for row in validation["attempts"]
    ]
    display(HTML(render_table(rows, caption="Measured first run and independent resume run")))

# %% [markdown]
# ## 7. Connect retrieval to the LAVA score
#
# LAVA averages semantic answer credit and evidence-page F1. Retrieval recall,
# MAP and nDCG are supporting diagnostics; they cannot substitute for that score.
# **There is no new 9B answer score in this notebook.** Its previous 87.02% local
# LAVA score was measured with oracle pages and must not be relabeled as end-to-end.
#
# Next: run the selected 9B reader on a prespecified retrieved-page budget, then
# score its saved answers with the unchanged Gemma judge and compare against the
# oracle result. Investigate missed evidence, including charts and scans, before
# adding a visual retriever. Finally freeze the complete pipeline, measure full
# runtime, generate all 624 test predictions, validate the submission, and verify
# authenticated Kaggle submission eligibility.
#
# References: [BM25 scoring](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html),
# [native PDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html),
# [LAVA evaluation](https://lava-workshop.github.io/#evaluation).
# %%
with logger.stage("07_record_next_milestone", heartbeat_seconds=15):
    display(
        HTML(
            render_table(
                [
                    {"Stage": "Oracle reader comparison", "State": "Complete · 4B, 9B, 27B"},
                    {"Stage": "Full-document text retrieval", "State": summary["status"]},
                    {"Stage": "9B with retrieved evidence", "State": "Next experiment"},
                    {
                        "Stage": "Complete test inference and Kaggle submission",
                        "State": "Pending complete-pipeline validation",
                    },
                ],
                caption="One coherent path toward a submission",
            )
        )
    )
logger.emit("retrieval.walkthrough.completed", question_count=summary["question_count"])
