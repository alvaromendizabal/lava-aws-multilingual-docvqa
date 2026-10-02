# Frontier research update — October 2026

This document records the public, employer-facing state of the LAVA reader/retrieval research after the September frontier work. It deliberately excludes private questions, raw generations, credentials, AWS locations, return bundles, and internal runner prompts.

The goal is to make the research **semi-reproducible**: enough model/evaluation identity, aggregate measurements, ablation design, and promotion logic are public to understand and audit the decisions, without publishing the complete competition implementation.

## Evaluation boundary

The reader-isolation panel contains **16 questions from five supplied training PDFs**: 15 Japanese and one Vietnamese. It is a reused development diagnostic, not an independent test set.

Local LAVA follows the published structure:

`overall = mean((answer_score + evidence_page_F1) / 2)`

Answer semantics use the project's pinned Gemma-3 1B judge. The organizer's exact prompt/runtime is unpublished, so these are local formula-based scores rather than official leaderboard scores.

Pinned semantic-judge identity:

- model: `google/gemma-3-1b-it`
- revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- contract: `6f3e8ea4f81bf99601d0a427bd541b27020e1668741d739ac97bde7e738222cc`

## Retrieval frontier

The retrieval program independently reconstructed lexical, multilingual dense, and page-image retrieval. Aggregate development results:

| Retriever | k | Evidence recall | Complete-evidence questions |
| --- | ---: | ---: | ---: |
| BM25 | 5 | 95.31% | 14/16 |
| multilingual E5 | 5 | 92.19% | 13/16 |
| ColQwen visual retrieval | 5 | 98.44% | 15/16 |
| BM25 + visual fusion | 5 | 98.44% | 15/16 |
| BM25 | 10 | 100.00% | 16/16 |
| ColQwen visual retrieval | 10 | 100.00% | 16/16 |

These numbers show that evidence availability is strong but not sufficient: reader/reasoning quality remains a material bottleneck.

## Reader frontier

The strongest verified reader on supplied gold evidence remains the pinned Qwen3.5-9B BF16 configuration:

- model: `Qwen/Qwen3.5-9B`
- revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- semantic answer score: **80.15%**
- evidence-page F1: **93.90%**
- local LAVA: **87.02%**
- valid responses: **16/16**

The strongest measured end-to-end retrieved-evidence system remains the two-pass citation-guided reread:

- semantic answer score: **67.65%**
- evidence-page F1: **88.33%**
- local LAVA: **77.99%**

That leaves roughly **9.0 percentage points** between the end-to-end system and the oracle-evidence 9B reader on this development diagnostic.

## 27B ceiling test

A later controlled experiment tested Qwen3.6-27B AWQ under the same 16-question oracle-evidence panel. Four reader strategies were evaluated:

- text direct
- fused text + page image
- verified/source-decomposition read
- selective adjudication

All four were then re-scored under the **same pinned semantic judge** used for the historical 9B result.

| Reader arm | Semantic answer | Grounding F1 | Local LAVA | Schema-valid rate |
| --- | ---: | ---: | ---: | ---: |
| 27B text direct | 44.38% | 79.17% | 61.77% | 81.25% |
| **27B fused direct** | **54.38%** | **90.63%** | **72.50%** | **100.00%** |
| 27B verified synthesis | 49.79% | 90.63% | 70.21% | 100.00% |
| 27B selective adjudication | 62.29% | 67.71% | 65.00% | 75.00% |
| Historical 9B fused reader | **80.15%** | **93.90%** | **87.02%** | **100.00%** |

Even an oracle selector that picks the best 27B arm after seeing the labels reaches only **79.58% local LAVA**, below the historical 9B reader.

### Decision

The one-shot Qwen3.6-27B branch is **not promoted**.

This is a valid negative result: the larger reader consumed more infrastructure complexity without improving the declared local metric. The project returns to the stronger 9B reader rather than continuing near-duplicate 27B prompt or deployment variants.

## What the 27B study taught us

1. **Vision helps:** fused direct materially beats text-only.
2. **Verbose verification is not automatically better:** the source-decomposition arm did not beat simple fused reading.
3. **Free-form adjudication is risky:** some answer accuracy improved, but grounding and schema reliability regressed.
4. **Model size is not the current ceiling:** the smaller 9B reader remains substantially stronger on the same semantic metric.
5. **Retrieval/perception remains the more plausible remaining gap:** the end-to-end system is still well below the oracle-evidence reader.

## Current ceiling-escape hypothesis

The next bounded reader experiment keeps the exact historical 9B revision fixed and tests two transferable mechanisms reported in recent public DocVQA work:

- **self-consistency:** multiple independent single-question samples with deterministic consensus;
- **active perception:** targeted crop/zoom views instead of only fixed full-page reading.

The planned controlled arms are:

| Arm | Capability |
| --- | --- |
| B | historical-style single deterministic reader |
| S | SC-8 independent samples + normalized majority consensus |
| A | single reader + targeted ROI/row/column zoom views |
| SA | consensus over SC-8 + active-perception candidates |

This experiment intentionally avoids publishing the private prompt wording or final voting heuristics. The public contract is the scientific comparison, not the competition-specific implementation.

Promotion requires all of:

- schema-valid rate at least 93.75%;
- improvement on at least three of five documents versus the current single-reader control;
- local semantic LAVA at least 3 points above the current single-reader measurement, or within 1 point of the historical 87.02% reader frontier, whichever is stricter.

If nothing promotes, the project will not repeatedly tune sampling temperatures, crop sizes, or vote weights. It moves to the next major capability: exhaustive/high-recall page screening plus explicit numerical/table reasoning.

## Next major modeling direction

The highest-priority retrieval/reasoning experiment is:

1. current retriever + fixed 9B control;
2. exhaustive lightweight page screening;
3. source-fact extraction;
4. targeted high-resolution reread only where necessary;
5. explicit numerical/table/multi-page computation;
6. the strongest validated 9B reader policy.

The project will evaluate answer quality and evidence grounding together. Retrieval recall alone does not qualify a challenger for promotion.

## Public-method research boundaries

The project has studied and independently adapted mechanisms from:

- public LAVA 2026 retrieval/reader implementations;
- exhaustive-page Qwen document-QA pipelines;
- HEAR-style structure-preserving extraction and verification;
- AdaDocVQA-style verified augmentation/task adaptation;
- recent DocVQA 2026 active-perception and self-consistency systems.

No competitor prediction files are imported. No public method is described as the winning system unless that placement is independently verified.

## Reproducibility boundary

This repository intentionally publishes:

- model and judge identities;
- aggregate metrics;
- experiment contracts;
- promotion/kill rules;
- negative results;
- public-method provenance;
- canonical executed notebooks for the earlier benchmark state.

It intentionally does **not** publish:

- private questions or reference answers;
- raw model generations;
- test predictions;
- private S3 object names;
- credentials;
- competition return bundles;
- exact internal prompts and orchestration details;
- model caches/checkpoints.

The six canonical notebooks remain checksum-bound to the research state they actually executed. New frontier work is documented separately until it earns a full notebook refresh. This avoids rewriting notebook manifests without legitimate re-execution.
