# LAVA Data Audit

Generated: 2026-09-02T00:13:32.057183+00:00

## Corpus integrity

- PDFs audited: **8**
- PDFs opened successfully: **8**
- PDF errors: **0**
- Total pages: **170**
- Exact duplicate PDF groups: **0**
- Cross-split exact duplicate groups: **0**

## Corpus composition

- Split counts: `{"test": 3, "train": 5}`
- Language counts: `{"ja": 7, "vi": 1}`
- Content classes: `{"native_text_dominant": 8}`
- Native-text page ratio: **0.994118**
- Embedded-image page ratio: **0.535294**

## Document scale

- Pages per PDF: min=4.0, median=19.0, p75=21.5, max=53.0
- PDF bytes: min=554546.0, median=1314648.0, p75=1676591.0, max=8791792.0

## CSV schemas

### `raw/kaggle/sample_submission.csv`

- Rows: **624**
- Columns: **3**
- Selected semantic columns: `{"answer": "answer", "document": null, "evidence": "evidence_page_number", "question": null}`
- Language counts: `{}`
- Answer types: `{"number": 204, "short_text": 420}`
- Evidence cardinalities: `{"1": 624}`
- Duplicate question rows: **0**
- Missing referenced documents: **0**

### `raw/kaggle/test.csv`

- Rows: **624**
- Columns: **5**
- Selected semantic columns: `{"answer": "answer_format", "document": "file_id", "evidence": null, "question": "question"}`
- Language counts: `{"ja": 587, "vi": 37}`
- Answer types: `{"short_text": 624}`
- Evidence cardinalities: `{}`
- Duplicate question rows: **0**
- Missing referenced documents: **0**

### `raw/kaggle/train.csv`

- Rows: **16**
- Columns: **7**
- Selected semantic columns: `{"answer": "answer_format", "document": "file_id", "evidence": "evidence_page_number", "question": "question"}`
- Language counts: `{"ja": 15, "vi": 1}`
- Answer types: `{"short_text": 16}`
- Evidence cardinalities: `{"1": 8, "2": 6, "3": 1, "4": 1}`
- Duplicate question rows: **0**
- Missing referenced documents: **0**

## Interpretation boundary

This phase audits source integrity, document scale, native-text availability, and label/schema structure. It does not use hidden test labels, does not tune a model, and does not claim that native text is sufficient for answering. Near-duplicate page detection, layout classification, OCR comparison, and multimodal modeling remain separate experiments so their contribution can be measured.

## Next research gate

Freeze a document-isolated validation protocol and implement the official answer-plus-evidence evaluator before training or selecting a retrieval or vision-language model.
