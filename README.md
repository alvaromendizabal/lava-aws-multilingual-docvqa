# LAVA — Multilingual Document Intelligence on AWS

[![CI](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml)

An evidence-grounded document-QA research project using open vision-language models, multilingual retrieval, reproducible AWS execution, and auditable evaluation.

**Current position:** data acquisition and audit, the 4B/9B/27B reader comparison, and full-document retrieval are complete. **Next:** evaluate the provisional 9B reader using retrieved pages. Complete-system performance and a Kaggle submission remain outstanding.

## Start in notebooks/

There is one canonical notebook folder: **[notebooks/](notebooks/)**. All five notebooks contain actual executed outputs. Read them in order; each can also run independently. Viewing them requires no GPU job, data download, or model-provider login.

| Notebook | What it explains | Current state |
| --- | --- | --- |
| [00 — Reproducibility and protocol](notebooks/00_reproducibility_and_protocol.ipynb) | Data already acquired, evaluation rules, project progress | Complete |
| [01 — Reader benchmark design](notebooks/01_oracle_reader_benchmark_design.ipynb) | Why these three model configurations were compared | Complete |
| [02 — Verified GPU execution](notebooks/02_verified_gpu_execution.ipynb) | Run coverage, cloud execution, checkpoints | Three full pilots verified |
| [03 — Model scaling and cost](notebooks/03_model_scaling_and_cost.ipynb) | Answer quality, evidence, latency, memory, cost | Three pilots scored; provisional 9B |
| [04 — Evidence retrieval](notebooks/04_evidence_retrieval.ipynb) | Search all PDF pages and measure retrieval | Complete; reader integration next |

In SageMaker Studio, open `/home/sagemaker-user/lava-aws-multilingual-docvqa/notebooks/`.
Do not navigate into an `artifacts/.../checkout` directory to read the project.
The obsolete validation checkouts and old installation bundles have been archived and removed from the active workspace.

## What has actually completed?

The full data audit verified **208 files, including 205 PDFs**: 16 training questions from five PDFs and 624 test questions from 200 PDFs. All available training labels are included in the pilots. Test answers are hidden; the sample submission is a template.

Each reader received the correct evidence pages. These are **oracle-evidence** results:

| Reader configuration | Semantic answer credit | Evidence-page F1 | Local LAVA score | Valid responses |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 4B | 50.63% | 97.02% | 73.82% | 16/16 |
| Qwen3.5 9B | **80.15%** | 93.90% | **87.02%** | 16/16 |
| Qwen3.8 27B NF4 | 70.98% | 89.73% | 80.36% | 15/16 |

The [published LAVA metric](https://lava-workshop.github.io/#evaluation) averages semantic answer credit and evidence-page F1 per question. Our local Gemma judge is pinned and passed 28 development controls; the organizer's exact prompt/runtime is unpublished. Local scores therefore carry an explicit organizer-parity limitation.

Keep **9B as the provisional reader**. The 27B experiment is already complete. Hardware, model generation, and precision differ, and five PDFs cannot establish general superiority. The invalid 27B response remains a counted failure.

The separate retrieval pilot searched **all 74 pages** in the five training PDFs, with zero extraction errors:

| Retrieved pages | BM25 evidence recall | Questions with all evidence found | Page-order all-evidence coverage |
| --- | ---: | ---: | ---: |
| 1 | 53.65% | 5/16 | 12.50% |
| 3 | 83.85% | 11/16 | 25.00% |
| 5 | **95.31%** | **14/16** | 37.50% |
| 10 | 100.00% | 16/16 | 56.25% |

At five pages, equal-document recall is 92.50% and all-evidence coverage is 75.00%. The initial run took **8.548 seconds**; an independent resume took **0.898 seconds**, reusing all five PDF extractions and 16 rankings. These are small training diagnostics. Retrieval coverage does not establish answer accuracy or held-out performance.

## Remaining delivery milestones

| Milestone | Work remaining | Completion evidence |
| --- | --- | --- |
| **1. Complete-system pilot** | Run 9B on retrieved pages; inspect missing evidence, invalid responses, and answer failures | Actual LAVA score, per-document/language/format diagnostics, runtime, memory, and verified resume |
| **2. Reproducible submission pipeline** | Package inference in Docker, verify the recorded single-GPU and two-hour limits, process all 200 test PDFs and 624 questions | Complete predictions with provenance and successful schema/ID/page validation |
| **3. Submission and portfolio release** | Confirm current Kaggle eligibility, submit if permitted, publish the final walkthrough and results/limitations | Accepted submission receipt or explicit eligibility blocker; reproducible tagged release |

The important missing experiment is milestone 1. Its failure analysis determines whether retrieval needs another iteration. There is no defensible percentage-complete or guaranteed finish date before that result. Additional larger models, optional ablations, and a web demo are not prerequisites for this delivery path.

## Run and resume

From the repository root:

```bash
# Verify or refresh the five canonical notebooks; reuse current outputs.
make notebooks

# Explicit tests, formatting, lint, types, compilation, and publication integrity.
make quality

# Inspect the CPU retrieval plan; no private download or paid compute.
make retrieval-preview

# Resume the existing retrieval evaluation from verified S3 checkpoints.
make retrieval-evaluate
```

Notebook outputs are published directly in `notebooks/`; checksum manifests live in `reports/notebook_execution/`. Completed publications can be reused even after cloning or merging. Changed inputs execute into staging first; interrupted publication resumes from the completed staging record. Failed execution preserves the last published notebook.

Reader checkpoints, retrieval checkpoints, and judge decisions persist in S3. Logs include UTC timestamps, elapsed time, progress, and heartbeats. Accepted SageMaker GPU jobs continue after browser disconnection; Studio CPU processes stop with the app and resume from saved checkpoints. Runtime limits still apply.

## Repository map

| Folder | Purpose |
| --- | --- |
| `notebooks/` | The five canonical, executed walkthroughs |
| `src/lava/` | Data, metrics, readers, retrieval, and observability implementation |
| `scripts/` and `Makefile` | Canonical commands |
| `configs/` | Pinned experiment, model, and evaluation contracts |
| `tests/` | Unit, integration, recovery, and publication checks |
| `pipelines/oracle_reader/` | Independent SageMaker GPU job entry point |
| `reports/` | Verified public aggregates, run history, and notebook manifests |
| `docs/` | Methodology and operating instructions |
| `infra/iam/` | Scoped submission-storage policy template |
| `artifacts/` | Ignored runtime files and caches; completed private work is preserved in S3 |

Empty scaffolding and duplicate notebook representations have been removed. Completed run history and source/model/data locks remain because they substantiate the results and enable resumption.

[Evaluation and recovery](docs/evaluation.md) · [Reader execution](docs/benchmark.md) · [Retrieval](docs/retrieval.md) · [Submission](docs/submission.md) · [Architecture](docs/architecture.md)
