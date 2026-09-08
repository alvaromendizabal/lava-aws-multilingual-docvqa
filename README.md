# LAVA — Multilingual Document Intelligence on AWS

[![CI](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml)

**Completed research benchmark · Python · Vision-language models · AWS · Reproducible evaluation**

How do model size, quantization, and compute cost affect multilingual document question answering—and how reliably can a simple retriever find the supporting pages? This project answers those questions with three measured reader configurations, full-document retrieval, a 1,582-configuration lexical search, and a visual retrieval challenger.

The portfolio deliverable is complete: verified data, frozen experiments, generated metrics, executed notebooks, visual analysis, and tested recovery. All notebook outputs are included. **An employer can review the work without an account, installation, or GPU.**

## Read the project

All six canonical notebooks live in [notebooks/](notebooks/). They can be read independently.

| Notebook | What to look for |
| --- | --- |
| [00 — Research overview](notebooks/00_reproducibility_and_protocol.ipynb) | Scope, verified headline results, and conclusions |
| [01 — Experiment design](notebooks/01_oracle_reader_benchmark_design.ipynb) | Comparable inputs, model configurations, and evaluation boundaries |
| [02 — Cloud execution](notebooks/02_verified_gpu_execution.ipynb) | Actual runs, checkpoints, logging, and recovery |
| [03 — Model quality and cost](notebooks/03_model_scaling_and_cost.ipynb) | Answer quality, citations, document effects, latency, memory, and cost |
| [04 — Evidence retrieval](notebooks/04_evidence_retrieval.ipynb) | Retrieval curves, 1,582 candidate configurations, visual challenger, and recovery |
| [05 — Retrieved-evidence extension](notebooks/05_end_to_end_system_evaluation.ipynb) | Implemented integration, actual measurement status, failure analysis and recovery |

**Short review:** read 00, then the results and conclusions in 03 and 04. Use 01 and 02 for methodological and engineering detail. No notebook needs to be run just to inspect the results.

## Measured findings

The data audit verified **208 files, including 205 PDFs**. The experiments cover **all 16 supplied training questions from five PDFs**. Fifteen questions are Japanese and one is Vietnamese. Hidden test answers are unavailable.

Each reader received the correct evidence pages and their native text. These are **oracle-evidence** measurements of answering and citation behavior:

| Reader configuration | Semantic answer credit | Evidence-page F1 | Local LAVA score | Valid responses |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 4B · BF16 | 50.62% | 97.02% | 73.82% | 16/16 |
| Qwen3.5 9B · BF16 | **80.15%** | 93.90% | **87.02%** | 16/16 |
| Qwen3.8 27B · NF4 | 70.98% | 89.73% | 80.36% | 15/16 |

**Finding:** 9B achieved the highest local score among these configurations. The 27B model did not improve this pilot; its invalid response remains a counted failure. Hardware, model generation, and precision differ, so parameter count alone cannot explain the result.

The [published LAVA metric](https://lava-workshop.github.io/#evaluation) averages semantic answer credit and evidence-page F1 per question. This implementation uses a pinned Gemma-3 1B judge that passed 28 development controls. The organizer's exact prompt/runtime is unpublished: these are local formula-based scores, with an explicit organizer-parity limitation.

The separate BM25 retrieval experiment searched **all 74 pages** in the training PDFs without using answer labels to rank pages:

| Page budget | Evidence recall | Questions with all evidence | Page-order all-evidence coverage |
| --- | ---: | ---: | ---: |
| 1 | 53.65% | 5/16 | 12.50% |
| 3 | 83.85% | 11/16 | 25.00% |
| 5 | **95.31%** | **14/16** | 37.50% |
| 10 | 100.00% | 16/16 | 56.25% |

**Finding:** lexical retrieval substantially improves evidence coverage over page order on this pilot. At five pages, equal-document recall is 92.50% and complete-evidence coverage is 75.00%, exposing variation hidden by the question average. The initial CPU run took **8.548 seconds**; a separate resumed run took **0.898 seconds**, reusing all five extractions and 16 rankings.

## Feature research and visual retrieval

The lexical search generated **1,089 BM25 scoring features and 493 fusion/exploration
policies**. Label-blind screening removed **349 duplicate rankings**, leaving 1,233
ranking signatures within the two search stages. These are alternative retrieval
configurations, not 1,582 inputs to a trained model. Free document-isolated selection
reduced recall@5 to 89.06%; conservative selection retained the original BM25 on all
five document folds. **No lexical challenger justified replacing the baseline.**

A completed, pinned **colSmol-500M page-image retrieval** experiment tested four
prespecified policies on the same 74 pages and 16 questions:

| Retrieval policy | Evidence recall@5 | Questions with all evidence |
| --- | ---: | ---: |
| BM25 baseline | 95.31% | 14/16 |
| colSmol-500M alone | 57.81% | 8/16 |
| Reciprocal rank fusion | 92.19% | 13/16 |
| Four BM25 pages + one novel visual page | **98.44%** | **15/16** |

The hybrid recovered one additional question on this development pilot. It has no
independent validation and is **not promoted to production**. The negative visual-only
result and rejected lexical challengers show why more complexity needs evidence.
[Notebook 04](notebooks/04_evidence_retrieval.ipynb) presents both searches;
[retrieval methodology](docs/retrieval.md) records provenance and limitations.

## What this demonstrates

- **Experimental judgment:** frozen model/data/prompt contracts, a common question set, counted failures, and explicit comparison limits.
- **Evaluation depth:** the LAVA formula, answer and citation diagnostics, document/language/format slices, exploratory uncertainty, retrieval ranking metrics, and measured resource use.
- **Reliable execution:** independent SageMaker GPU jobs, per-question S3 checkpoints, checksum verification, and recovery from interrupted work.
- **Readable evidence:** six executed notebooks with explanations, tables, charts, UTC timestamps, elapsed time, progress, and heartbeats.
- **Software quality:** unit and real-kernel integration tests, recovery tests, formatting, lint, types, and CI. Tests validate correctness; benchmark metrics describe model behavior.

## Scope and limitations

This release completes a **component-level research benchmark**. Five labeled PDFs support descriptive findings; they do not establish held-out performance, language-wide accuracy, or state-of-the-art task performance. The readers are frozen pretrained models; this project does not claim to train a new foundation model.

The 87.02% reader score uses oracle pages. Retrieval was evaluated separately, so no end-to-end answer score is claimed. Native-text retrieval also does not establish visual understanding of every chart or scanned page.

Reader evaluation with retrieved pages, a deployed application, full test inference, and Kaggle submission are **optional extensions outside this completed release**. No submission or leaderboard result is claimed. There are no additional experiments required to review or use this portfolio benchmark.

The retrieved-page pipeline and user-operated CSV export remain documented in
[Notebook 05](notebooks/05_end_to_end_system_evaluation.ipynb) as optional extensions.
The pending integrated GPU attempt was stopped during the September 8 closeout;
no end-to-end answer score is claimed. No further compute is required for this
portfolio release. The notebook defaults only inspect published results, and Kaggle
upload remains a user decision. See [closeout evidence](docs/closeout.md).

## Reproduce and inspect

In the configured Studio environment, use `/home/sagemaker-user/lava-aws-multilingual-docvqa`. Public analysis uses the pinned Python 3.12 environment. Register its notebook kernel once with `uv run --frozen python -m ipykernel install --user --name lava --display-name "Python (LAVA)"`. Then `make notebooks` verifies and reuses current outputs or refreshes changed notebooks; `make quality` runs the explicit quality gate. Neither command launches a GPU.

Notebook publication verifies source, input, and output hashes; interrupted writes recover from completed staging records, and failed execution preserves the previous publication. Private source data, responses, and model/retrieval/judge checkpoints persist in S3. GitHub preserves code, public aggregates, and executed notebooks.

| Location | Purpose |
| --- | --- |
| `notebooks/` | The complete employer reading path |
| `src/lava/`, `scripts/`, `configs/` | Implementation, canonical commands, and frozen contracts |
| `tests/` | Correctness, integration, recovery, and publication checks |
| `reports/` | Measured public results, provenance, and notebook manifests |
| `pipelines/oracle_reader/`, `infra/iam/` | GPU job entry point and scoped storage policy |
| `docs/` | Detailed methodology and reproduction instructions |

[Evaluation and recovery](docs/evaluation.md) · [Architecture](docs/architecture.md) · [Reader execution](docs/benchmark.md) · [Retrieval](docs/retrieval.md)
