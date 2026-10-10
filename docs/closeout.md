# Project closeout · document intelligence and public evidence

This release presents the document-intelligence system I built, its historical evaluation record, and a runnable browser demonstration of evidence retrieval and citation inspection. Public delivery and scientific answer completeness are reported separately.

## Delivered scope

| Component | Recorded state |
|---|---|
| Research pipeline | Page retrieval, multimodal reading, structured responses and citation evaluation |
| Original routed development candidate | **82.68% local LAVA** on the reused 16-question/five-PDF panel |
| Best recorded submission | **0.49 public / 0.53 private** |
| Later recovered diagnostic | **0.48 public / 0.49 private**; no improvement |
| Recovered prediction acceptance | **622/624** structurally accepted non-abstaining predictions; two unresolved |
| Recovery evidence | **1,081** unchanged inference checkpoints verified in the historical AWS closeout |
| Public demo | [Document Desk](https://alvaro-document-evidence.tartmacaw2.chatgpt.site): actual lexical retrieval, extraction, cited support and abstention |
| Review path | [Case study](case_study.md), [review guide](reviewer_guide.md), [reproduction guide](reproducibility.md) |

The historical closeout completed 14 audit tasks, verified saved/reopened notebook outputs and a versioned backup, and made no new inference calls or submissions. This public demo release is separate from that recorded execution.

## Research limits preserved

The two unresolved answers still require actual source support. Schema-valid placeholders do not resolve them. The historical best contained two template-derived values without verified support; the later diagnostic contained two compatibility abstentions.

The original 82.68% local route result does not transfer automatically to the later recovered composite. Repeated use of five documents limits generalization confidence; additional independent document-level evidence would be needed for a stronger performance claim.

## Public demonstration boundary

Document Desk uses authored Japanese and Vietnamese text with BM25 and sentence extraction. It performs real local retrieval and validates copied support and physical-page citations. It does not run the research vision-language models, OCR pages or establish a new official score. The lexical support measure is not calibrated confidence.

The repository publishes reusable code, tests, configuration, executed notebooks, aggregate results and synthetic examples. Private documents, predictions, raw generations, exact routing rules, model caches, credentials and cloud locations remain excluded. No production service-level guarantee is claimed.

[Sanitized submission evidence](submission_closeout.md) · [Architecture](architecture.md)
