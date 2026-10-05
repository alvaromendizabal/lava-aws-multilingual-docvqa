# Project maturity and public roadmap — October 2026

This page is the stable public milestone ledger for LAVA. It records measured research that can be reviewed without exposing private competition artifacts.

## Verified milestones

| Milestone | Result | Decision |
| --- | --- | --- |
| Data audit | 208 raw files, 205 PDFs verified | complete |
| Qwen3.5-9B supplied-evidence reader | **87.02% local LAVA** | reader frontier |
| BM25 full-PDF retrieval | 95.31% recall@5 | strong baseline retained |
| visual / lexical retrieval hybrid | **98.44% recall@5** | strong exploratory evidence |
| Qwen retrieved-evidence first pass | 67.57% local LAVA | baseline system |
| Qwen citation-guided reread | **77.99% local LAVA** | incumbent system |
| larger one-shot reader study | best arm below verified reader frontier | stopped |
| self-consistency / active perception | improved same-run control | failed promotion contract |
| exhaustive screening / explicit reasoning | strong numeric slice | specialist signal |
| Gemma heterogeneous reader | complementary error profile | routing candidate |
| nested document-disjoint heterogeneous route | **82.68% local LAVA** | validated challenger |

All local quality measurements use the frozen development panel and pinned semantic-judge contract documented in this repository.

## Project maturity

### Research
**Mature.** Baselines, ablations, model-family comparisons, retrieval studies, negative experiments, promotion rules, and held-out-document validation are all represented in public evidence.

### Software
**Mature research codebase.** The repository includes typed schemas, reusable packages, unit/integration tests, linting, typing, CI, executed notebooks, and explicit configuration contracts.

### Cloud execution
**Production-oriented research workflow.** GPU runs use deterministic identities, resumable checkpoints, bounded runtime/cost controls, resource telemetry, and recovery-first failure handling.

### Reproducibility
**Publicly semi-reproducible by design.** Public source, configs, executed notebooks, aggregate evidence, and quality gates are available; private competition outputs and exact routing rules are intentionally excluded.

## What this project demonstrates to an employer

- Ability to turn an ambiguous multimodal problem into measurable subsystems.
- Strong retrieval/RAG and vision-language engineering.
- Discipline around group-aware validation and leakage prevention.
- Comfort with GPU constraints, cloud orchestration, and failure recovery.
- Willingness to kill expensive ideas when evidence is weak.
- Ability to preserve a clean public portfolio surface while operating a more sensitive private evaluation layer.

## Public roadmap

Future public work should only be added when it creates durable evidence.

1. Expand independent document-level validation beyond the current small labeled panel.
2. Benchmark stronger dense/visual retrieval under the same frozen selection contract.
3. Evaluate adaptive multimodal reading with failure-safe, resumable execution.
4. Refresh canonical notebooks only when a new stable milestone materially changes the project story.
5. Keep external competition outcomes separate from local research metrics unless the comparison is explicitly like-for-like.

## Publication boundary

The public repository does not mirror private test predictions, private source documents, raw private model generations, credentials, exact routing rules, or private cloud return bundles.

That boundary keeps the project useful to employers while avoiding a public dump of competition-sensitive artifacts.

[Portfolio](portfolio.md) · [Review guide](reviewer_guide.md) · [Reproducibility](reproducibility.md) · [Frontier research](frontier_research_update.md) · [Architecture](architecture.md)
