# Frontier reader benchmark

## Purpose

This benchmark reconstructs and compares complete reader systems rather than treating retrieval metrics as the final objective. It keeps the original 9B pipeline as a control, independently rebuilds the documented public 27B retrieval-and-reader route, and tests two source-verification extensions. All six arms use the same 16-question development panel and the repository's frozen local LAVA metric.

The benchmark is intentionally separate from production submission generation. It does not train weights, submit to Kaggle, choose a leaderboard blend, or promote a configuration automatically.

## Fixed systems

| Arm | System | Role |
| --- | --- | --- |
| A | Qwen3.5-9B, BM25 top five | Existing compact control |
| B | Qwen3.5-9B, BM25 top ten | Context-coverage ablation |
| C | Qwen3.6-27B NF4, reconstructed BM25Plus/E5/adaptive route | Public-method reference |
| D | Qwen3.6-27B NF4, lexical/visual adaptive route | Project adaptation |
| E | D plus bounded evidence decomposition | Source-verification experiment |
| F | D plus independent text/image views and conflict rereading | Cross-modal adjudication experiment |

The maximum is 96 final predictions and 144 unique generation requests. Each raw generation is content-addressed and immutable. Reference answers are unavailable to the generation process and are opened only after all predictions are persisted.

## Verified frontier state

The retrieval work established the following development diagnostics:

| Retrieval policy | Candidate pages | Evidence recall | Complete-evidence questions |
| --- | ---: | ---: | ---: |
| BM25 | 5 | 95.31% | 14/16 |
| ColQwen visual retrieval | 5 | 98.44% | 15/16 |
| BM25 | 10 | 100.00% | 16/16 |
| ColQwen visual retrieval | 10 | 100.00% | 16/16 |

These values diagnose evidence availability; they are not answer scores or official competition results.

The first complete reader field run passed both model contracts:

- pinned Qwen3.5-9B in BF16;
- pinned Qwen3.6-27B in NF4;
- strict loading diagnostics;
- one real multimodal compatibility generation per model;
- all 28 public judge acceptance probes.

It persisted all 32 control answers and the first 27B reference answer before stopping. The stop was not a GPU or model failure. The 27B model emitted exact decimal page strings (`["11", "12"]`), while the first parser required JSON integers and aborted on the first structurally invalid output. No local benchmark score was published from that incomplete run.

## Resumability correction

The corrected runner separates immutable raw model generations from parser results. A parser-only correction can therefore reparse completed model outputs instead of paying to regenerate them. Migration is accepted only when the legacy request envelope, value checksum, model pins, package versions, system prompt, model runtime, and GPU contract match.

The correction normalizes only serialization-equivalent physical page values: exact positive decimal strings become integers. It rejects floats, booleans, signed strings, ranges, prose, zero, duplicates, and pages outside the supplied context. Answer content is never rewritten.

The model-preflight identity is now independent of parser and reporting code. Completed 9B and 27B compatibility receipts remain reusable when the model runtime and hardware contract are unchanged. The benchmark still stops after three consecutive invalid final answers in one arm and never silently resamples a cached failure.

## Interpretation boundary

This is a previously examined 16-question development panel: 15 Japanese questions and one Vietnamese question across five PDFs. It can compare mechanisms and expose failures, but it cannot establish held-out multilingual performance or state-of-the-art status. The official Kaggle score remains unverified until authenticated, read-only score receipts are captured. No claim of beating the winner is made before a complete scored run and a separate confirmation evaluation.
