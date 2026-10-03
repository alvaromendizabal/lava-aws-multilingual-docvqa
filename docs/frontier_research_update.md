# Frontier research update — October 2026

This page records the public-safe state of the LAVA reader, retrieval, reasoning, and validation program after the executed-notebook benchmark. It excludes private questions, raw generations, test predictions, credentials, cloud locations, exact routing rules, and competition-specific orchestration.

## Evaluation boundary

The development panel contains **16 questions from five supplied training PDFs**: 15 Japanese and one Vietnamese. Local LAVA follows the published structure `mean((answer_score + evidence_page_F1) / 2)` under the pinned Gemma-3 1B semantic judge (`dcc83ea841ab6100d6b47a070329e1ba4cf78752`). These are frozen local research measurements; external competition evaluation remains separate.

## Current research frontier

| View | Local LAVA | Interpretation |
| --- | ---: | --- |
| Qwen3.5-9B on supplied gold evidence | **87.02%** | reader-isolation diagnostic |
| Qwen3.5-9B retrieved-evidence two-pass system | **77.99%** | measured incumbent |
| Heterogeneous routed system, nested held-out-document evaluation | **82.68%** | validated challenger |

The routed result is an out-of-fold estimate: the candidate family is fixed before each held-out document is evaluated, and selection uses only the other documents.

## Retrieval frontier

| Retriever | k | Evidence recall | Complete-evidence questions |
| --- | ---: | ---: | ---: |
| BM25 | 5 | 95.31% | 14/16 |
| multilingual E5 | 5 | 92.19% | 13/16 |
| ColQwen visual retrieval | 5 | **98.44%** | **15/16** |
| BM25 + visual fusion | 5 | **98.44%** | **15/16** |
| BM25 | 10 | 100.00% | 16/16 |
| ColQwen visual retrieval | 10 | 100.00% | 16/16 |

Evidence availability is strong but not sufficient: answer composition, numerical reasoning, grounding discipline, and reader specialization remain material bottlenecks.

## Research sequence

### Larger-reader study

A controlled larger-reader branch tested multiple one-shot strategies under the same semantic contract. The strongest arm reached **72.50% local LAVA**, below the verified 9B reader. Decision: **do not promote**. The project stopped spending on near-duplicate larger-model prompt variants.

### Self-consistency + active perception

A later Qwen9B study tested repeated sampling and targeted crop/zoom perception.

| Arm | Local LAVA |
| --- | ---: |
| same-run deterministic single reader | 82.34% |
| self-consistency ensemble | **83.48%** |
| active-perception single reader | 81.29% |
| self-consistency + active perception | **83.48%** |

The best ensemble improved over its same-run control but failed the multi-document promotion contract, so the branch was stopped rather than micro-tuned.

### Exhaustive screening + explicit reasoning

| Arm | Local LAVA |
| --- | ---: |
| exhaustive screening + direct reader | 52.46% |
| screening + targeted reread | 56.77% |
| screening + explicit reasoning | **69.21%** |
| incumbent two-pass system | **77.99%** |

No global arm earned promotion. The explicit-reasoning arm was nevertheless **perfect on all four numeric development questions** for both answer quality and grounding, exposing a useful specialization signal.

### Heterogeneous Gemma reader

A compressed Gemma 4 12B reader was evaluated as a deliberately different model family.

| Gemma condition | Semantic answer | Grounding F1 | Local LAVA |
| --- | ---: | ---: | ---: |
| supplied gold evidence | 71.88% | 90.63% | 81.25% |
| retrieved evidence, direct | 44.79% | 86.46% | 65.63% |
| retrieved evidence, self-citation reread | **72.92%** | 83.33% | **78.13%** |

Gemma did not earn standalone promotion. Its value was complementary error structure relative to Qwen, motivating a heterogeneous candidate family instead of another same-family ensemble.

## Document-disjoint heterogeneous validation

The project froze a small candidate family and ran nested leave-one-document-out selection.

- incumbent two-pass Qwen system: **77.99%**
- out-of-fold heterogeneous routed system: **82.68%**
- delta: **+4.69 percentage points**
- documents improved: **2**
- documents regressed: **0**
- exploratory document-cluster bootstrap delta interval: approximately **+1.79 to +16.18 points**
- heterogeneous route selected in four of five outer folds

The exact routing rule is intentionally private. The public contract is the validation design: candidate family frozen before the held-out document is scored, selection without that document's labels, and every held-out document retained in the aggregate.

## Why this progression matters

The research program follows a production-oriented decision discipline:

1. establish strong simple baselines;
2. measure complete-system behavior;
3. reject expensive complexity when evidence is weak;
4. test qualitatively different capabilities instead of endless prompt variants;
5. retain negative experiments;
6. identify specialization signals;
7. combine genuinely complementary model families;
8. validate routing on held-out documents before treating it as a challenger.

## Cloud reliability work

The private AWS inference system uses deterministic model/data/source contracts, content-addressed artifacts, per-question checkpoints, resumable GPU runs, bounded runtime/cost controls, heartbeats, resource telemetry, strict parsing, process isolation/cleanup, representation-parity checks, and failure packaging that preserves completed work.

The private 624-question workflow is active in AWS and intentionally not mirrored into Git. The public repository exposes reusable architecture and aggregate research evidence, not competition outputs.

## Reproducibility boundary

Published: model/revision identities, judge/evaluation identities, aggregate metrics, experiment contracts, promotion/kill rules, negative results, executed benchmark notebooks, and public-safe machine-readable artifacts.

Not published: private questions/answers, test predictions, raw generations, exact routing logic, credentials/cloud object names, private checkpoints/model caches, or competition-specific return bundles.
