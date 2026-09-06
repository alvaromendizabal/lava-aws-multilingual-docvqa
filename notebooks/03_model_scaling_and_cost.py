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
# # 03 — Which reader should we use?
#
# **Question:** does the 27B candidate improve document question answering enough
# to justify its memory, latency and cost? Start with the evidence below. A larger
# model is a candidate to measure; it is not automatically the better system.
#
# Run these cells from top to bottom. They use saved, checksum-verified results;
# opening or rerunning this notebook does not launch GPUs or download models.
# Each step logs UTC time and elapsed time, with a heartbeat during longer work.
#
# ## 1. Load the experiment record
#
# A smoke test answers one question to prove that loading, generation and storage
# work. A full pilot answers all 16 supplied labeled questions. Semantic scoring
# is a separate step applied to saved answers; it does not rerun the reader.
# %%
import json
from pathlib import Path

from IPython.display import HTML, display

from lava.evaluation.reporting import load_report, render_report
from lava.evaluation.walkthrough import (
    TABLE_STYLE,
    candidate_assessment,
    comparison_tables,
    render_table,
    training_rates,
)
from lava.notebook_support import find_repo_root
from lava.readers.evaluation_contract import load_evaluation_contract
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = find_repo_root(Path.cwd())
logger = RuntimeEventLogger("notebook.reader_comparison")
with logger.stage("01_load_verified_results", heartbeat_seconds=15):
    report = load_report(ROOT)
    contract = load_evaluation_contract(ROOT)
    pricing = json.loads((ROOT / "reports/aws/training_prices.json").read_text())
    tables = comparison_tables(report, training_rates(pricing))
    display(HTML(TABLE_STYLE + render_table(tables["coverage"], caption="What has completed?")))
# %% [markdown]
# ## 2. Understand what these data can tell us
#
# These 16 questions from five PDFs are **all supplied training labels**, not an
# arbitrary limit of the experiment. Most questions are Japanese; one is Vietnamese.
# Several questions share a document, so they are not 16 independent documents.
# This is a descriptive development pilot, not held-out proof of generalization.
#
# The reader receives the known evidence pages plus their extracted native text.
# This isolates answering and citation behavior. It does not test finding evidence
# in a whole PDF. Full-document retrieval is the next distinct experiment.
# %%
with logger.stage("02_explain_coverage", heartbeat_seconds=15):
    coverage = [
        {"Dimension": "Questions", "Count": contract["question_count"]},
        {"Dimension": "Documents", "Count": contract["document_count"]},
        *[
            {"Dimension": f"Language: {key}", "Count": value}
            for key, value in contract["language_counts"].items()
        ],
        *[
            {"Dimension": f"Answer format: {key}", "Count": value}
            for key, value in contract["answer_format_counts"].items()
        ],
    ]
    display(HTML(render_table(coverage, caption="The frozen labeled pilot")))
# %% [markdown]
# ## 3. Compare answer quality with the published LAVA formula
#
# The organizer metric averages **semantic VQA** (answer credit) and **evidence-page
# F1** (citation credit), then averages across questions:
#
# $$LAVA = \frac{1}{N}\sum_{i=1}^{N}\frac{VQA_i + EvidenceF1_i}{2}.$$
#
# Our pinned Gemma-3 1B judge implements the local semantic evaluation. List
# answers can receive partial credit. Missing/invalid outputs remain in the
# denominator. These are local formula-based results, not organizer-server scores.
# A missing score means **not evaluated**, not zero.
#
# Valid output measures whether a response follows the required format. It does
# not establish that the answer is correct. Evidence F1 here evaluates citations
# among supplied oracle pages; it must not be presented as retrieval performance.
# %%
with logger.stage("03_compare_quality", heartbeat_seconds=15):
    display(
        HTML(
            render_table(
                tables["quality"],
                percent_columns=(
                    "Semantic VQA",
                    "Evidence F1",
                    "Local LAVA overall",
                    "Valid output",
                    "Abstention",
                ),
                caption="Quality on the same labeled questions · higher is better except abstention",
            )
        )
    )
# %% [markdown]
# ## 4. Check whether the gain is consistent across documents
#
# Question averages weight documents with more questions more heavily. The table
# below also gives each PDF equal weight and counts improvements and regressions.
# An overall average can conceal a document on which the larger model does worse.
#
# Differences are **challenger minus baseline**, in percentage points. Five PDFs
# provide weak uncertainty estimates: bootstrap intervals are exploratory, and
# repeated model comparisons are not adjusted for multiple testing. These results
# guide the next experiment; they do not certify broad model superiority.
# %%
with logger.stage("04_compare_documents", heartbeat_seconds=15):
    names = {run["job_name"]: run["label"] for run in report["runs"]}
    paired = [
        {
            "Comparison": f"{names[pair['challenger_job']]} minus {names[pair['baseline_job']]}",
            "Metric": pair["metric_label"],
            "Question delta (pp)": 100 * pair["question_mean_delta"],
            "Document delta (pp)": 100 * pair["mean_delta"],
            "PDFs improved / tied / regressed": (
                f"{pair['documents_improved']} / {pair['documents_tied']} / "
                f"{pair['documents_regressed']}"
            ),
        }
        for pair in report["semantic_document_comparisons"]
    ]
    display(HTML(render_table(paired, caption="Compatible semantic evaluations only")))
# %% [markdown]
# ## 5. Account for speed, memory and compute cost
#
# Generation mean and p95 describe per-question generation, excluding setup.
# Billable time includes the measured SageMaker job duration that AWS bills.
# Capacity waiting, image download, weight loading, generation and upload are
# different phases; a slow wall-clock run need not mean a slow reader.
#
# Estimated compute cost = billable seconds / 3600 × the dated regional Training
# price. It excludes Studio, storage, logs, data transfer, tax and discounts.
# It is neither an invoice nor an account spending cap. A smoke run is excluded
# from this comparison because setup cost dominates its one-question workload.
# %%
with logger.stage("05_compare_systems", heartbeat_seconds=15):
    display(
        HTML(render_table(tables["systems"], caption="Measured full-pilot systems performance"))
    )
    print(f"Price snapshot verified at {pricing['checked_at_utc']}")
    print(pricing["scope"])
# %% [markdown]
# ## 6. Understand what changed between candidates
#
# **B means billions of model parameters.** More parameters can increase capacity,
# but memory, training recipe, quantization and implementation also affect results.
# The 4B and 9B readers use Qwen3.5 with BF16 weights. The 27B candidate uses
# Qwen3.8 with NF4 (4-bit) quantization to fit the proven single-GPU path.
#
# Therefore this is a comparison of configured systems, not an experiment that
# isolates model size. GPU hardware differs too. Frozen model/code revisions,
# identical questions, prompting and judge contracts make the result auditable.
# %%
with logger.stage("06_show_lineage", heartbeat_seconds=15):
    display(
        HTML(render_table(tables["lineage"], caption="Exact models and code used for inference"))
    )
# %% [markdown]
# ## 7. Inspect the visual report and remaining errors
#
# The report contains per-document and per-format charts, uncertainty intervals,
# citation precision/recall, exact-page-set rate, latency and the full run history.
# A score-gap breakdown tells us where credit was lost; it does not tell us the
# cause. Inspect actual errors before changing prompts or retrieval.
# %%
with logger.stage("07_render_visual_report", heartbeat_seconds=15):
    display(HTML(render_report(report)))
# %% [markdown]
# ## 8. Make the next decision
#
# 1. Read the measured 27B-versus-9B result below. Only complete, compatible
#    semantic evaluations enter this decision; a one-question smoke cannot rank it.
# 2. If 27B gains answer credit, inspect document regressions and extra latency.
#    A small gain on five PDFs does not automatically justify the larger reader.
# 3. Keep the best justified reader as a provisional baseline. Evaluate page
#    retrieval over whole documents, then rerun the reader with retrieved pages.
# 4. Measure the complete pipeline against the organizer's two-hour budget on
#    one A100 40GB. These generation-only timings do not prove that requirement.
# 5. Generate all 624 test predictions with non-oracle provenance; validate the
#    submission schema and exact test coverage with `docs/submission.md`.
#
# The test set has no public answer labels. Never report its quality from training
# scores. The normal competition deadline was May 31, 2026; authenticated late
# submission eligibility must be checked before claiming a Kaggle submission.
#
# **What you can show an employer:** a reproducible experiment, honest limits,
# quality/cost tradeoffs, durable artifacts, tested failure recovery and a clear
# next hypothesis. A larger parameter count alone is not evidence of state-of-the-art
# task performance. Official links: [LAVA](https://lava-workshop.github.io/) and
# [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
# %%
with logger.stage("08_assess_candidate", heartbeat_seconds=15):
    assessment = candidate_assessment(
        report, "qwen35_9b_fused_direct", "qwen38_27b_nf4_g5_fused_direct"
    )
    display(HTML(render_table([assessment], caption="What does the 27B comparison support?")))
logger.emit(
    "walkthrough.completed",
    verified_runs=len(report["runs"]),
    complete_current_pilots=sum(run["complete"] for run in report["current_models"]),
    new_gpu_jobs=0,
    reader_inference_repeated=False,
)
