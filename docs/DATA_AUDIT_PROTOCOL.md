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

## Source-consistency follow-up

The September 10, 2026 follow-up tests a narrow hypothesis: an unresolved question
may reflect unread source content, a stale file, or a question-to-document mismatch.
It does not optimize retrieval weights or answer prompts on test questions.

1. Restore the frozen question mapping, original PDF versions, prepared page
   counts and cached extraction results; compare their exact hashes.
2. Inspect current host notices and the signed-in data interface. Distinguish
   a successful download checksum from a live preview or a failed download.
3. Fix a diagnostic term query before searching all supplied test-document text.
   Retain every matching page, provenance and the number of image-only pages.
4. Reuse the assigned documents' saved native/OCR text. Verify every object byte
   hash, version, page number and expected coverage; inspect representative images.
5. Record candidate mappings as hypotheses. Never substitute another PDF, invent
   an answer, or convert missing evidence into a no-answer label without authority.
6. Checkpoint the diagnostic source, fixed queries, exact object references,
   results, limitations and resource state. Publish only aggregate findings.

The run verified 200 extraction envelopes, 4,698 page records and 63 assigned-PDF
OCR envelopes. Three local search smoke checks passed. A page-field mismatch was
caught and fixed before the search completed; no cloud inference was involved.
Search of the restored corpus took 1.771 seconds; restoring and checksum-verifying
the 200 cached extractions took 266.731 seconds. Transfer and browser time are
separate from search time. Per-document receipts allow transfer resumption.

The outcome is a source-clarification blocker, with 622 complete predictions
preserved. Native absence is not visual absence: 798 corpus pages have no native
text. Cached OCR for both assigned documents supplied additional evidence, but
neither a replacement source nor an intentional-unanswerable policy was verified.
The full private audit also records a prior narrative object's checksum-metadata
discrepancy; it does not treat that metadata as proof or rewrite the old object.
Frozen input and extraction hashes were verified independently.
