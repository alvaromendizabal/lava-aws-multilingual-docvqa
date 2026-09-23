# Frontier reproduction program

This project treats the existing LAVA pipeline as a **control**, not as an architecture that must be preserved. The research goal is to independently reconstruct documented public methods, measure them under one protocol, then combine only components that add complementary value. No competitor submission files or private predictions are imported.

## What is already measured

On the 16-question / 5-document development diagnostic, the original BM25 top-5 retrieval reaches **95.31% evidence recall and 14/16 complete-evidence questions**. Multilingual E5 alone is weaker at top-5 (**92.19%, 13/16**). ColQwen visual retrieval and BM25+visual RRF reach **98.44%, 15/16** at top-5. At top-10, both BM25 and visual retrieval recover all required evidence on **16/16** questions. These are development retrieval metrics, not official competition scores.

The important implication is that retrieval is no longer treated as a fixed five-page bottleneck: candidate recall and citation precision are measured separately. Broad candidate retrieval can reach 16/16 while the reader must still cite only the pages that actually support its answer.

## Reader reproduction benchmark

The fixed six-arm benchmark compares:

| Arm | Reader | Context / reasoning |
| --- | --- | --- |
| A | Qwen3.5-9B BF16 | BM25 top-5, direct |
| B | Qwen3.5-9B BF16 | BM25 top-10, direct |
| C | Qwen3.6-27B NF4 | documented UIT-style BM25Plus + E5 adaptive routing |
| D | Qwen3.6-27B NF4 | lexical + ColQwen adaptive context |
| E | Qwen3.6-27B NF4 | D + source-grounded decomposition |
| F | Qwen3.6-27B NF4 | independent text/image views + conflict reread |

Both pinned readers have passed strict GPU loading and tiny generation probes on an L40S. The latest field run persisted all 16 outputs for A, all 16 for B, and the first C generation. It stopped because C serialized evidence page numbers as digit strings. That is a representation issue, not a new answer-generation failure: the v4 parser losslessly converts digit-only physical page identifiers to integers while continuing to reject floats, decimals, booleans, duplicates and unavailable pages.

The v4 cache contract also separates deterministic **generation identity** from downstream parser/report implementation. Parser-only fixes revalidate the saved raw response rather than paying to resample the model. Model-load receipts are similarly isolated from downstream orchestration changes.

## Reproduction boundaries

The public 2026 UIT code is treated as a documented reference implementation, but its final competition placement has not been independently verified. Arm C is therefore described as a **public-method adaptation**, not a certified winner reproduction. Using published pretrained weights also does not reproduce foundation-model training.

The project will not claim a leaderboard improvement from local development metrics. Official score receipts remain separate, and the latest read-only Kaggle score audit could not authenticate from the AWS workspace.

## Next research gates

1. Finish and score all six reader arms using the already cached A/B outputs and the cached first C response.
2. Attribute errors to retrieval, reading/reasoning, evidence selection, or source inconsistency.
3. Add task-specific training only after building an audited, document-disjoint multilingual validation/training corpus.
4. Select an answer ensemble only on held-out evidence; do not learn weights on the 16-question diagnostic.
5. Generate a new full-test candidate only after the end-to-end system beats the frozen control under the declared local protocol.

The repository intentionally keeps private questions, reference answers, test predictions, credentials, raw return bundles and model caches out of Git.
