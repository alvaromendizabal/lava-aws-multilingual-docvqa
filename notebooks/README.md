# Executed research notebooks

All six canonical notebooks include verified execution outputs. **Open and read; no setup or GPU is required.**

| Notebook | Purpose |
| --- | --- |
| [00 — Research overview](00_reproducibility_and_protocol.ipynb) | Scope, evaluation contract, headline benchmark results |
| [01 — Experiment design](01_oracle_reader_benchmark_design.ipynb) | Comparable model inputs and research methodology |
| [02 — Cloud execution](02_verified_gpu_execution.ipynb) | AWS execution, checkpoints, logging and recovery |
| [03 — Model quality and cost](03_model_scaling_and_cost.ipynb) | Reader comparisons, uncertainty and resource tradeoffs |
| [04 — Evidence retrieval](04_evidence_retrieval.ipynb) | Lexical, structural and visual retrieval research |
| [05 — Complete system](05_end_to_end_system_evaluation.ipynb) | Retrieved-evidence answering and citation-guided rereading |

**Short employer review:** 00 → 05 → 03 → 04.

The notebook set is intentionally a checksum-bound research snapshot. Later October experiments—including self-consistency, active perception, exhaustive reasoning, heterogeneous Gemma reading, and document-disjoint routing validation—are published separately in the [frontier research update](../docs/frontier_research_update.md) and [aggregate research artifacts](../research/README.md).

This separation is deliberate: later metrics are not retroactively inserted into executed notebooks without rerunning those notebooks against the new source/input state.

## What Notebook 05 established

The integrated benchmark measures the complete full-PDF BM25 → Qwen3.5-9B path and a deterministic citation-guided second read.

On the frozen development panel:
- first pass: **67.57% local LAVA**
- citation-guided reread: **77.99%**
- improvement: **+10.42 percentage points**

The later document-disjoint heterogeneous challenger reaches **82.68% local LAVA** and is documented outside the canonical notebook snapshot.

## Reproduction

In Studio this folder is:

`/home/sagemaker-user/lava-aws-multilingual-docvqa/notebooks/`

For reproduction:

```bash
make notebooks
make quality
```

Those commands verify/reuse outputs or refresh changed public inputs and create no GPU job.

See [Portfolio overview](../docs/portfolio.md) for the quickest project review.
