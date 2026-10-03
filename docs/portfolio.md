# Portfolio overview — LAVA multilingual document intelligence

## Executive summary

I built LAVA as an applied ML research system for multilingual document question answering on AWS. The project spans data auditing, multimodal retrieval, reader benchmarking, evidence grounding, semantic evaluation, document-disjoint model selection, resumable GPU inference, and reproducible publication.

The current public research frontier is a **heterogeneous routed document-QA system validated at 82.68% local LAVA with nested held-out-document selection**, improving a measured two-pass Qwen incumbent at 77.99%.

The repository is intentionally semi-reproducible: it publishes architecture, source, model identities, validation logic, aggregate evidence, and executed notebooks while withholding private questions, test predictions, raw generations, exact competition routing rules, and credentials.

## Why this project is technically interesting

### 1. It measures the whole system

A strong reader is not enough. On supplied gold evidence, Qwen3.5-9B reaches **87.02% local LAVA**; the first retrieved-evidence system falls to **67.57%**. That gap forced the project to treat retrieval, context construction, reasoning, and grounding as first-class research problems.

### 2. It turns citations into a model-control mechanism

A deterministic second pass rereads only the first-pass model's valid citations. The same 9B reader improves from **67.57% to 77.99%** without changing weights.

### 3. It tests genuinely different failure modes

The research program includes lexical retrieval, multilingual dense retrieval, page-image late interaction, visual/lexical fusion, larger multimodal readers, self-consistency, targeted crop/zoom perception, exhaustive page screening, explicit numerical/table reasoning, and a second multimodal reader family. Branches are stopped when they fail promotion gates.

### 4. It validates model routing instead of trusting a pooled blend

A different multimodal reader produced complementary errors but did not win as a standalone model. Instead of publishing a post-hoc mixture, I froze a small candidate family and used nested leave-one-document-out selection.

Result:
- incumbent: **77.99%**
- routed out-of-fold estimate: **82.68%**
- **+4.69 percentage points**
- 2 documents improved
- 0 documents regressed

The exact route is private; the validation design is public.

### 5. It is engineered for interrupted cloud work

Long-running GPU research is treated as a resumable pipeline with per-question durable checkpoints, deterministic lineage contracts, content-addressed inputs, exact read-back verification, bounded cost/runtime, resource heartbeats, explicit retry semantics, process isolation, and strict final coverage gates.

## Key measured results

| Area | Result |
| --- | --- |
| Data integrity | 208 raw files / 205 PDFs audited |
| Reader frontier | 87.02% local LAVA on supplied gold evidence |
| Two-pass incumbent | 77.99% local LAVA |
| Document-disjoint heterogeneous challenger | **82.68% local LAVA** |
| Best measured recall@5 | **98.44%** |
| Lexical/fusion retrieval configurations audited | **1,582** |
| Executed research publication | **6 checksum-bound notebooks + required CI quality gate** |

## Technologies demonstrated

**ML / NLP / multimodal:** Python, PyTorch, Transformers, Qwen multimodal models, Gemma multimodal models, BM25, multilingual embedding retrieval, visual late-interaction retrieval, semantic evaluation, ensemble/routing validation.

**Cloud / engineering:** AWS SageMaker, S3, deterministic batch jobs, checkpoint stores, hash-based provenance, resource telemetry, bounded GPU execution, CI.

**Research practice:** ablations, negative-result retention, held-out-document selection, bootstrap uncertainty, fixed promotion gates, leakage-aware evaluation, reproducibility contracts.

## Recommended review

1. [README](../README.md)
2. [Notebook 00](../notebooks/00_reproducibility_and_protocol.ipynb)
3. [Frontier research update](frontier_research_update.md)
4. [Validated routing & inference engineering](heterogeneous_routing_update.md)
5. [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb)
6. [Architecture](architecture.md)
7. [Machine-readable frontier evidence](../research/README.md)

## Publication boundary

Published: source code, frozen public configs, executed notebooks, aggregate metrics, model/revision identities, validation methodology, negative experiments, and system architecture.

Private: exact test predictions, raw generations, exact routing rule, private documents where restricted, credentials, cloud object locations, and model caches/checkpoints.

That boundary keeps the project credible, reviewable, and employer-facing while preserving the private inference implementation.
