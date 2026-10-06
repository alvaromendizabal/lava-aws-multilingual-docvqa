# LAVA — Multilingual Document Intelligence on AWS

[![CI](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml)

**PDF question answering · Multimodal retrieval · Vision-language models · AWS GPU inference**

I built an **Applied ML research system** that answers questions from Japanese and Vietnamese PDFs and returns the physical pages supporting each answer. My work covers evidence retrieval, structured multimodal reading, evaluation, model comparison, and recovery from interrupted GPU runs.

The practical problem is finding an answer in a long document while keeping its source inspectable. LAVA separates retrieval errors, reading errors, and citation errors so a model change can be evaluated at the stage it affects.

**Start here:** [Case study](docs/case_study.md) · [Run the public example](examples/README.md) · [Executed system notebook](notebooks/05_end_to_end_system_evaluation.ipynb)

![Synthetic Japanese and Vietnamese examples showing answers and physical-page citations](examples/evidence_example.svg)

*Illustration uses authored synthetic documents and supplied responses. The runnable example checks the real public parser and submission validator; it does not run a model or measure accuracy.*

## Results and what they mean

| Evidence | Result | Evaluation scope |
| --- | ---: | --- |
| Best recorded competition submission | **0.49 public / 0.53 private** | Organizer scores; refreshed October 6, 2026 |
| Later recovered diagnostic | 0.48 public / 0.49 private | Organizer scores; did not improve the recorded best |
| Original heterogeneous route | **82.68% local LAVA** | Document-held-out selection; reused **16 questions / 5 PDFs** |
| Citation-guided reread | **77.99% local LAVA** | Same **16 questions / 5 PDFs**; post-hoc development finding |
| Best measured evidence recall@5 | **98.44%** | Retrieval only; same **16 questions / 5 PDFs** |
| Preserved inference checkpoints | **1,081** | Byte-identical in the verified AWS closeout |

The local route result applies to the original frozen route, **not the later recovered prediction composite**. Local scores and official scores use different evaluation populations and runtimes and should not be compared as if interchangeable.

The historical best CSV has 624 structurally valid rows, including two template-derived values without verified support. The recovered candidate has **622 structurally accepted predictions out of 624**, with two unresolved answers; its diagnostic CSV uses two compatibility abstentions. Structural acceptance is not a count of correct answers. The latest closeout verified existing artifacts and submitted nothing new. [Results, provenance, and completion status →](docs/submission_closeout.md)

## Engineering decisions

- **Measure retrieval before replacing the reader.** Compare BM25, multilingual dense embeddings, and ColQwen page-image retrieval against physical-page labels. A broader audit covered 1,582 retrieval configurations with fold-isolated selection.
- **Test a second read before scaling the model.** Citation-guided rereading raised the local end-to-end score from 67.57% to 77.99% on the reused development panel. Larger readers, extra pages, and repeated sampling did not consistently clear promotion gates.
- **Select complementary readers by document.** Evaluate a frozen Qwen/Gemma routing family while holding out whole documents. The 82.68% local result is exploratory evidence on a small reused panel, not independent confirmation of generalization.
- **Preserve expensive work.** Hash model, data, source, and prediction identities; checkpoint each question; capture failure diagnostics; and verify saved notebooks and versioned backups. Runtime failure and a negative model result remain separate outcomes.

[Architecture](docs/architecture.md) · [Research decisions](docs/frontier_research_update.md) · [Reliability and recovery](docs/reliability_recovery.md)

## Run a public example

Python 3.12, no GPU, account, model download, or third-party package required:

```bash
python3 examples/run_demo.py
python3 examples/verify_submission_closeout.py
```

The first command checks supplied Japanese/Vietnamese responses and rejects an invalid citation using the public implementation. The second verifies the aggregate closeout report and recomputes its comparisons. Neither command reproduces private inference or an official score.

For the full repository quality gate and the six frozen research notebooks:

```bash
uv sync --frozen --group judge
make quality
make notebooks
```

This installs the pinned research environment, including its CPU judge dependencies. See [reproduction levels and prerequisites](docs/reproducibility.md) before model-backed evaluation.

## Development benchmarks

All tables in this section use the same **16-question / five-PDF** development panel and pinned local semantic judge. These measurements are **not presented as organizer-server-identical competition scores**.

| Supplied-evidence reader | Semantic answer | Evidence F1 | Local LAVA | Valid responses |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 4B · BF16 | 50.62% | 97.02% | 73.82% | 16/16 |
| Qwen3.5 9B · BF16 | 80.15% | 93.90% | **87.02%** | 16/16 |
| Qwen3.8 27B · NF4 | 70.98% | 89.73% | 80.36% | 15/16 |

| Retrieved-evidence system | Semantic answer | Evidence F1 | Local LAVA |
| --- | ---: | ---: | ---: |
| BM25 → Qwen3.5-9B | 48.90% | 86.25% | 67.57% |
| Citation-guided Qwen3.5-9B reread | 67.65% | 88.33% | **77.99%** |

Fixed BM25 achieved **95.31% recall@5**; the strongest measured visual/lexical policy reached **98.44%**. More retrieved pages did not reliably improve answer quality. [Retrieval study →](docs/retrieval.md)

## Executed research notebooks

These six notebooks preserve the benchmark state they actually executed. Later aggregate research and closeout evidence are published separately, without rewriting historical outputs.

| Notebook | Evidence |
| --- | --- |
| [00 — Research overview](notebooks/00_reproducibility_and_protocol.ipynb) | Scope, contracts, measured results, and conclusions |
| [01 — Experiment design](notebooks/01_oracle_reader_benchmark_design.ipynb) | Comparable reader inputs and configurations |
| [02 — Cloud execution](notebooks/02_verified_gpu_execution.ipynb) | AWS jobs, hashes, checkpoints, and recovery |
| [03 — Quality and cost](notebooks/03_model_scaling_and_cost.ipynb) | Quality, latency, memory, and cost tradeoffs |
| [04 — Evidence retrieval](notebooks/04_evidence_retrieval.ipynb) | Full-PDF retrieval and fold-level decisions |
| [05 — End-to-end evaluation](notebooks/05_end_to_end_system_evaluation.ipynb) | Retrieved-evidence answers and citation-guided rereading |

## Scope and limitations

This is an implemented research pipeline, with testing, typing, CI, and AWS execution evidence. Production deployment and service-level performance are not claimed. The small development panel has been reused across experiments and has limited Vietnamese coverage. Broader independent document-level validation remains necessary.

The private full-test and competition layer is intentionally excluded from public Git. Public code, configuration, synthetic examples, executed development notebooks, aggregate results, and verification commands are available. Exact routing rules, private predictions, raw generations, model caches, credentials, cloud object locations, and private return bundles remain excluded.

The competition objective remains unresolved: the recorded best is below the leader, and the strict complete-answer check remains at 622/624. The public evidence supports an engineering case study and a bounded account of the research; it does not claim winner reproduction or a completed 624-answer system.

[Technical review guide](docs/reviewer_guide.md) · [Reproducibility](docs/reproducibility.md) · [Public evidence](research/README.md) · [Project status](docs/closeout.md)
