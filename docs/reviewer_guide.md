# Employer review guide

This repository is organized so a reviewer can assess the project at three levels: product/ML impact, research quality, and engineering execution. The public surface intentionally exposes enough evidence to audit the work without publishing private competition predictions, exact routing rules, credentials, or cloud object locations.

## 5-minute review

1. Read the [engineering case study](case_study.md) for the problem, constraints, decisions, results, and tradeoffs.
2. Scan the project summary and executive visual in the [README](../README.md).
3. Open [Notebook 00](../notebooks/00_reproducibility_and_protocol.ipynb) for the research contract and [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb) for the complete system.
4. Review the [frontier research update](frontier_research_update.md) to see what was promoted, rejected, and why.

A reviewer should come away with four facts quickly:

- this is a complete multilingual document-intelligence system, not a single model notebook;
- retrieval, multimodal reading, evidence grounding, semantic scoring, and routing are measured separately;
- model and system changes are promoted through explicit validation gates rather than anecdotal examples;
- AWS execution is designed around resumability, lineage, observability, and bounded failure recovery.

## 15-minute technical review

| Capability | Evidence |
| --- | --- |
| Multimodal document AI | [Architecture](architecture.md), [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb) |
| Retrieval / RAG | [Retrieval research](retrieval.md), [Notebook 04](../notebooks/04_evidence_retrieval.ipynb) |
| Model evaluation | [Notebook 03](../notebooks/03_model_scaling_and_cost.ipynb), [Evaluation](evaluation.md) |
| Group-aware validation | [Frontier update](frontier_research_update.md), nested held-out-document result |
| AWS / GPU engineering | [Notebook 02](../notebooks/02_verified_gpu_execution.ipynb), [Architecture](architecture.md) |
| Reproducibility | [Reproducibility guide](reproducibility.md), notebook manifests, pinned configs |
| Software quality | CI, unit/integration tests, Ruff, mypy, publication checks |

## 30-minute deep review

Follow one decision end to end:

1. **Problem framing:** a question must be answered from a complete PDF with physical-page evidence.
2. **Evidence discovery:** compare lexical, multilingual dense, and page-image retrieval.
3. **Reader isolation:** compare multimodal readers on supplied evidence.
4. **Complete-system evaluation:** introduce retrieval and measure the quality lost outside the oracle setting.
5. **Reasoning refinement:** test citation-guided rereading, active perception, explicit reasoning, and model-family diversity.
6. **Selection discipline:** reject branches that fail predeclared gates.
7. **Held-out validation:** evaluate a frozen heterogeneous candidate family with document-disjoint selection.
8. **Operationalization:** persist checkpoints, hashes, run metadata, resource telemetry, and failure bundles.

## What the project demonstrates

### Applied ML research
- Controlled ablations instead of one-off prompt changes.
- Separation of reader quality, retrieval quality, and end-to-end quality.
- Negative results retained as evidence.
- Explicit promotion and kill criteria.
- Document-disjoint validation for model-family routing.

### ML engineering
- Typed schemas and strict structured-output parsing.
- Model/data/source contracts and content-addressed artifacts.
- Per-question checkpoints and resumable execution.
- Resource-aware GPU workflows with bounded runtime and cost.
- CI-backed notebooks and public reports.

### Product thinking
- Physical-page citations are treated as a first-class output, not an afterthought.
- Accuracy is balanced with evidence quality, latency, memory, reliability, and recoverability.
- Public claims are limited to measurements that can be traced to a frozen evaluation contract.

## Claims policy

The headline metrics in this repository are **local research measurements** on the released development material under a pinned implementation of the published LAVA metric structure. They are not presented as organizer-server-identical competition scores.

The public repository intentionally excludes:
- private test questions and predictions;
- raw model generations from private evaluation;
- exact competition-specific routing rules;
- credentials and cloud object names;
- private return bundles and model caches.

That boundary is deliberate: the goal is to make the engineering and research methodology inspectable without publishing competition-sensitive artifacts.

## Recommended interview discussion

The strongest technical discussions usually start with one of these:

- Why did a larger reader fail to beat the 9B reader?
- Why did retrieval recall improve without guaranteeing answer quality?
- How was heterogeneous routing validated without leaking the held-out document?
- How does the checkpoint design prevent expensive GPU work from being lost?
- Why are the executed notebooks frozen instead of retroactively rewritten after later experiments?
