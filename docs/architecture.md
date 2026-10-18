# Document intelligence architecture

The maintained reference path is **full-PDF text extraction → label-blind BM25 page ranking → the selected Qwen3.5-9B reader → physical-page citation validation → citation-guided reread → independent evaluation**.

| Component | Source | Contract |
|---|---|---|
| Retrieval | [pipeline](../src/lava/retrieval/pipeline.py) · [lexical ranking](../src/lava/retrieval/lexical.py) | Stable physical page IDs; extraction and ranking bound to source hashes |
| Reading | [Qwen reader](../src/lava/readers/qwen35.py) · [typed schemas](../src/lava/readers/schemas.py) | Label-free reader inputs; reference answers visible only to evaluation |
| Output validation | [structured output](../src/lava/readers/structured_output.py) | Exact generation parsed independently; citations restricted to supplied pages |
| Second pass | [refinement](../src/lava/readers/refinement.py) | Reuse valid model-cited pages without consulting gold labels |
| Recovery | [checkpoints](../src/lava/readers/checkpoints.py) | Compatible identities and durable completion; incomplete work is not a cache hit |
| Evaluation | [metric](../src/lava/evaluation/metric.py) · [system evaluation](../src/lava/evaluation/system.py) | Invalid answers remain in the denominator; answer and grounding credit remain distinct |

The browser demonstrator executes lexical extraction over authored Japanese and Vietnamese documents. It is not a hosted neural reader. Its export is independently replayed by [the verifier](../tools/replay_public_demo.mjs).

Project AWS resources are retired. The source retains reusable execution contracts; it does not provide live cloud object locations, private documents, trained artifacts, full-test generations or exact final routing rules.
