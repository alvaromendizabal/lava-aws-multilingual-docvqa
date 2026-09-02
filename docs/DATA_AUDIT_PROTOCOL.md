# Data-audit protocol

The audit is deliberately model-free. It verifies the Kaggle-to-S3 data foundation,
profiles the three CSV schemas without retaining raw question or answer values in the
public report, and scans each PDF sequentially from S3.

For each PDF it records cryptographic integrity, split, language identifier, page count,
native-text coverage, embedded-image coverage, text/word/block totals, page geometry,
PDF metadata, processing time, and any read error. A durable JSON-lines checkpoint is
written after every document and synchronized to S3, making the scan resumable.

The audit detects exact duplicate PDFs by SHA-256. Near-duplicate pages, layout classes,
OCR alternatives, and page-image embeddings remain separate experiments because those
methods require additional assumptions and must be evaluated independently.

No hidden labels are used. No model is trained or selected during this phase.
