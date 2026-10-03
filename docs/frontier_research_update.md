# Frontier research update — October 2026

This document records the public, employer-facing state of the LAVA reader and retrieval research after the original September benchmark release. It intentionally excludes private questions, raw generations, test predictions, credentials, cloud object names, internal runner prompts, and exact competition routing logic.

The public goal is **semi-reproducibility**: enough model identity, aggregate measurements, validation design, engineering constraints, and promotion logic are available to understand why the system changed without publishing a turnkey solution.

## Evaluation boundary

The main development panel contains **16 supplied questions across five PDFs**: 15 Japanese and one Vietnamese. Local LAVA follows the published structure:

`overall = mean((answer_score + evidence_page_F1) / 2)`

Answer semantics use the project's pinned Gemma-3 1B judge. The organizer's exact prompt/runtime is not reproduced here, so the measurements below are local development scores, not official hidden-test results.

Pinned semantic-judge identity:

- model: `google/gemma-3-1b-it`
- revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- contract: `6f3e8ea4f81bf99601d0a427bd541b27020e1668741d739ac97bde7e738222cc`

## Retrieval program

The project independently reconstructed lexical, dense multilingual, visual page retrieval, hybrid fusion, exhaustive screening, and document-structure signals.

Representative development results:

| Retriever | k | Evidence recall | Complete-evidence questions |
| --- | ---: | ---: | ---: |
| BM25 | 5 | 95.31% | 14/16 |
| multilingual E5 | 5 | 92.19% | 13/16 |
| ColQwen visual retrieval | 5 | 98.44% | 15/16 |
| BM25 + visual fusion | 5 | 98.44% | 15/16 |
| BM25 | 10 | 100.00% | 16/16 |

The important conclusion was not simply “increase recall.” Several higher-context strategies failed to improve answer quality. Evidence selection, multimodal reading, numerical reasoning, and reader specialization remained material bottlenecks.

## Reader frontier

The strongest pinned reader on supplied gold evidence remains Qwen3.5-9B BF16:

- model: `Qwen/Qwen3.5-9B`
- revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- semantic answer: **80.15%**
- evidence-page F1: **93.90%**
- local LAVA: **87.02%**
- schema-valid responses: **16/16**

The original end-to-end two-pass system measures:

- semantic answer: **67.65%**
- evidence-page F1: **88.33%**
- local LAVA: **77.99%**

Its fixed policy is BM25 retrieval followed by Qwen3.5-9B and a second read over the model's own validated citations.

## Larger-reader scaling study

A controlled larger-Qwen branch tested multiple direct and verification-style reader strategies under the same development evaluator.

The larger branch increased deployment complexity but did not earn promotion. The project therefore retained the empirically stronger 9B reader instead of optimizing around parameter count.

This branch produced an important engineering decision: **scale only when the measured system improves**, not because a model is newer or larger.

## Self-consistency and active perception

The next reader study evaluated:

- a deterministic single-reader control;
- SC-8 self-consistency;
- targeted crop/zoom active perception;
- a combined self-consistency + active-perception arm.

Self-consistency improved the local score relative to the experiment's single-reader control, but the gain was concentrated in too few documents to pass the promotion rule. Fixed crop/zoom did not add an independent improvement.

Decision: no promotion and no micro-tuning loop over temperatures, crop sizes, or voting weights.

## Exhaustive screening and explicit reasoning

A later experiment combined:

1. exhaustive lightweight page screening;
2. targeted reread;
3. explicit numerical/table/multi-page reasoning.

Aggregate local LAVA:

| Arm | Local LAVA |
| --- | ---: |
| Exhaustive screening + direct reader | 52.46% |
| Screening + targeted reread | 56.77% |
| Screening + explicit reasoning | 69.21% |
| Two-pass Qwen baseline | **77.99%** |

No full-system challenger promoted.

A useful specialist signal did emerge: the explicit-reasoning arm was perfect on the four numeric development questions. That result informed later work on specialization, but the project did not promote a post-hoc numeric route on the same data.

## Heterogeneous Gemma reader

The project then introduced a genuinely different reader family:

- `google/gemma-4-12B-it-qat-w4a16-ct`
- W4A16 compressed checkpoint
- pinned vLLM inference environment

Gemma's standalone BM25 self-citation system measured **78.13% local LAVA**, essentially competitive with the 77.99% Qwen end-to-end baseline but with a very different error profile.

By answer format, Gemma was particularly strong on strings and numbers and materially weaker on unordered lists.

The important result was therefore **complementarity**, not a standalone replacement.

## Document-disjoint routing validation

A compact routing family was frozen before validation. Reader selection used only public-safe question metadata such as language and answer format; reference answers and gold evidence never entered inference.

Nested held-out-document selection produced:

| System | Local LAVA | Held-out documents improved | Held-out documents regressed |
| --- | ---: | ---: | ---: |
| Two-pass Qwen baseline | 77.99% | — | — |
| Heterogeneous routed challenger | **82.68%** | **2** | **0** |

The routed system therefore improves the document-disjoint development measurement by **4.69 percentage points**.

Four of five folds selected the same compact policy family. A separate full-panel diagnostic was directionally consistent, but the document-disjoint result is the primary validation evidence.

This is still a small supplied development set. The project does not treat it as hidden-test evidence.

## Full-test inference engineering

After the route was frozen, the project shifted from model research to reliable execution on a single NVIDIA L4-class GPU.

Engineering work included:

- exact historical representation recovery and byte-for-byte page-render validation;
- immutable route/model/renderer contracts;
- reuse of historical first-pass Qwen checkpoints;
- per-question answer checkpoints;
- deterministic second-pass replay;
- constrained-memory Qwen inference;
- exact-output parity before CPU weight offload;
- vLLM batch-size parity testing before throughput promotion;
- bounded execution with resource heartbeats;
- process-group cleanup so inference-engine descendants cannot survive a timeout;
- fail-closed stale-GPU-worker ownership checks.

The Qwen routed slice has been fully checkpointed. The Gemma routed slice is being completed through the same resumable contract.

The repository intentionally publishes these engineering principles and aggregate results without publishing private test predictions or the exact routing implementation.

## Why this matters beyond the benchmark

The later work demonstrates a general production ML pattern:

1. establish a strong single-model baseline;
2. test larger/newer models instead of assuming superiority;
3. measure complementary residual structure;
4. constrain the routing family;
5. validate selection on held-out groups;
6. freeze the routing contract;
7. optimize runtime only behind numerical-parity gates;
8. checkpoint inference at the smallest economically useful unit;
9. turn infrastructure failures into regression tests.

This is the same discipline required for reliable multimodel systems in production.

## Public-method boundaries

The project has studied and independently adapted ideas from public document-intelligence work, including:

- lexical and multilingual dense retrieval;
- page-image late-interaction retrieval;
- exhaustive page screening;
- structure-preserving extraction;
- active perception;
- self-consistency;
- explicit numerical reasoning;
- heterogeneous multimodal readers.

No competitor prediction files are imported. Public methods are used as research inspiration and independently reimplemented.

## Reproducibility boundary

Public:
- model/revision identities where useful;
- evaluation contracts;
- aggregate metrics;
- validation methodology;
- negative results;
- architecture;
- engineering/recovery principles.

Private:
- private questions and reference answers;
- raw model generations;
- test predictions;
- cloud object names;
- credentials;
- exact internal prompts;
- exact competition router/recovery heuristics;
- execution return bundles.

The six canonical notebooks remain checksum-bound to the research state they actually executed. Later frontier work is documented separately until it earns a legitimate notebook refresh.

[Validated routing and inference engineering](heterogeneous_routing_update.md) · [Architecture](architecture.md) · [Research artifacts](../research/README.md)
