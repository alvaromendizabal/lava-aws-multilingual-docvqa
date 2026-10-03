# Verified milestones and active frontier — October 2026

This page is the public milestone ledger for the LAVA project. It records completed, measured research and separates the executed notebook snapshot from later frontier experiments.

## Current measured milestones

| Milestone | Result | Decision |
| --- | --- | --- |
| Data audit | 208 raw files, 205 PDFs verified | complete |
| Qwen3.5-9B supplied-evidence reader | **87.02% local LAVA** | reader frontier |
| BM25 full-PDF retrieval | 95.31% recall@5 | strong baseline retained |
| visual / lexical retrieval hybrid | **98.44% recall@5** | exploratory, not default |
| Qwen retrieved-evidence first pass | 67.57% local LAVA | baseline system |
| Qwen citation-guided reread | **77.99% local LAVA** | incumbent system |
| larger one-shot reader study | best arm 72.50% | not promoted |
| self-consistency / active perception | best arm 83.48% in same-run study | did not pass promotion contract |
| exhaustive screening / explicit reasoning | 69.21% overall; perfect numeric slice | specialist signal only |
| Gemma 4 heterogeneous reader | 78.13% standalone self-cite | complementary, not standalone champion |
| nested document-disjoint heterogeneous route | **82.68% local LAVA** | validated challenger |

All local quality measurements use the frozen 16-question development panel from five PDFs and the pinned semantic-judge contract.

## What the project demonstrates

### Applied ML research
- compares model families under a consistent metric;
- keeps negative experiments in the record;
- uses predeclared promotion gates;
- distinguishes reader quality, retrieval quality, and complete-system behavior;
- uses held-out documents to validate heterogeneous model routing.

### Document intelligence
- multilingual PDF parsing;
- physical-page evidence retrieval;
- native text + page-image reasoning;
- structured answers with page citations;
- table/numeric reasoning experiments;
- visual retrieval and targeted perception.

### Cloud ML engineering
- AWS batch inference;
- resumable per-question checkpoints;
- content-addressed artifacts;
- exact model/data/source lineage;
- bounded spend/runtime controls;
- heartbeat/resource observability;
- strict failure packaging and recovery.

### Reproducible publication
- six executed notebooks;
- machine-readable aggregate result files;
- static and interactive reports;
- unit/integration tests, lint, typing, and CI;
- explicit public/private artifact boundary.

## Canonical notebook snapshot

The six notebooks under `notebooks/` are the executed benchmark snapshot. Their manifests bind the source, public inputs, and completed outputs they actually ran.

The earlier publication gate completed with:
- **546 tests passed**
- Ruff across 149 files
- mypy across 84 source files
- all six notebooks executed end to end
- no notebook error/stderr outputs in the publication set

Later October frontier experiments are published in `research/` and `docs/frontier_research_update.md` rather than inserted into older notebooks without rerunning them.

## Current private completion stage

The private 624-question inference workflow is implemented as a resumable AWS pipeline with strict final coverage, answer-schema, and evidence-page validation.

Private test predictions and exact competition routing mechanics are intentionally excluded from Git. The public repository focuses on reusable engineering and validated aggregate research.

## Remaining public milestones

1. finish the current private full-test inference path;
2. freeze the final structural candidate and provenance bundle;
3. run external competition evaluation separately from local research metrics;
4. feed that external result back into the next research decision;
5. refresh the canonical employer-facing notebook only when there is a meaningful new stable milestone.

[Portfolio overview](portfolio.md) · [Frontier research](frontier_research_update.md) · [Architecture](architecture.md)
