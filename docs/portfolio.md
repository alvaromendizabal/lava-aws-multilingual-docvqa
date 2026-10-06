# Portfolio overview

## LAVA — multilingual document intelligence on AWS

**Problem:** answer questions from complete Japanese and Vietnamese PDFs while returning the physical pages that support each answer.

**System:** full-document retrieval → multimodal reading → structured answer generation → citation validation → semantic + grounding evaluation → document-disjoint model selection.

**Engineering posture:** production-style contracts, resumable GPU execution, content-addressed artifacts, typed schemas, observability, CI, and public/private artifact separation.

## Measured highlights

| Result | Measurement | Why it matters |
| --- | ---: | --- |
| Strongest supplied-evidence reader | **87.02% local LAVA** | isolates reader capability when evidence is known |
| Two-pass retrieved-evidence incumbent | **77.99% local LAVA** | complete end-to-end system with citation-guided rereading |
| Heterogeneous routed challenger | **82.68% local LAVA** | nested held-out-document selection across model families |
| Best measured recall@5 | **98.44%** | page-image/lexical retrieval recovers evidence missed by a lexical baseline |
| Retrieval configurations audited | **1,582** | broad search with fold-isolated selection, not manual cherry-picking |

These are local research measurements on the released development material under a pinned implementation of the published LAVA scoring structure. External competition scoring is intentionally kept separate.

## What I built

### Retrieval
- BM25 and multilingual dense retrieval.
- Page-image retrieval with ColQwen-style late interaction.
- Fusion and adaptive evidence discovery.
- Full-PDF physical-page accounting.
- Fold-isolated policy selection.

### Multimodal readers
- Qwen-family vision-language readers.
- A heterogeneous Gemma-family reader.
- Structured outputs with physical-page citations.
- Citation-guided rereading.
- Active-perception and explicit-reasoning experiments.

### Evaluation
- Answer semantics and evidence grounding scored separately.
- Frozen evaluation and semantic-judge contracts.
- Per-question, per-document, language, and answer-format slices.
- Document-disjoint routing validation.
- Promotion gates and negative-result retention.

### AWS / MLOps
- SageMaker GPU execution.
- Per-question checkpoints and resume.
- Exact model/data/source lineage.
- Bounded runtime and cost gates.
- Heartbeats, resource telemetry, process cleanup, and failure packaging.
- Versioned artifact storage and checksum verification.
- Failure classification that separates infrastructure/runtime faults from valid negative experiments.
- Resume-first recovery: completed inference survives downstream notebook, packaging, or runtime failures.

## Research decisions that matter

The project records why ideas were stopped as carefully as why ideas were promoted.

- A substantially larger one-shot reader did **not** beat the verified 9B reader under the pinned semantic contract.
- More retrieval pages did not automatically improve answer quality because irrelevant context can dilute grounding.
- Self-consistency improved a same-run control but did not satisfy the cross-document promotion contract.
- Explicit reasoning showed a strong numeric specialization signal without earning global promotion.
- A heterogeneous reader was only valuable after its complementary error pattern survived document-disjoint routing validation.

That progression demonstrates model selection discipline rather than model-shopping.

## Reviewer path

### 5 minutes
- [README](../README.md)
- [Employer review guide](reviewer_guide.md)
- [Notebook 00](../notebooks/00_reproducibility_and_protocol.ipynb)

### 15 minutes
- [Notebook 05 — complete system](../notebooks/05_end_to_end_system_evaluation.ipynb)
- [Notebook 04 — retrieval](../notebooks/04_evidence_retrieval.ipynb)
- [Frontier research update](frontier_research_update.md)

### Deep technical review
- [Architecture](architecture.md)
- [Evaluation](evaluation.md)
- [Reliability and recovery engineering](reliability_recovery.md)
- [Reproducibility](reproducibility.md)
- [Machine-readable research evidence](../research/README.md)

## Public/private boundary

The repository is designed to be inspectable without publishing the competition-sensitive layer.

Published:
- reusable source code;
- frozen configs and contracts;
- executed development notebooks;
- aggregate metrics;
- validation design;
- public-safe experiment history;
- reliability and observability abstractions.

Not published:
- private test predictions;
- raw private generations;
- exact competition routing rules;
- credentials or cloud object names;
- private return bundles and model caches.

This boundary is part of the engineering design, not missing documentation.
