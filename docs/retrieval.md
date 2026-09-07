# Full-document evidence retrieval

Notebook 04 asks whether the system can find the pages needed to answer a question.
The earlier reader comparison received gold evidence pages. Its 9B score remains
an oracle reader measurement until we actually run 9B on retrieved evidence.

## Completed experiment

The fixed BM25 baseline searched 74 pages from all five labeled training PDFs.
All pages had some native text and no extraction error occurred. These facts do
not establish that diagrams, charts or scans are fully readable as text.

At k=5, BM25 has 95.31% question-average evidence recall and 87.50% complete-evidence
coverage (14/16 questions), compared with 45.31% recall and 37.50% complete coverage
for page order. Equal-document BM25 recall is 92.50%; complete coverage is 75.00%.
The remaining incomplete cases are in doc-01 and the sole Vietnamese example,
doc-05. At k=10 all evidence is retrieved on this pilot. This is not an estimate
of test-set performance or a reason to claim perfect question answering.

The first CPU run took 8.548 seconds, including verified source reads, extraction,
ranking, scoring and checkpoint persistence. A second process took 0.898 seconds
and reused all five extractions and 16 rankings. Attempt JSONL logs include UTC,
stage and total elapsed time, per-document/per-question progress and 15-second
heartbeats during longer stages. Completed and failed attempt logs are archived
with read-back verification. No new compute resource was created.

## Ranking without labels

`RetrievalQuery` contains only question ID, document ID and question text. The
ranker receives this question and every physical page of its PDF. Answers,
answer formats, languages and gold evidence are not features. The evaluator
introduces reference labels only after the rankings have been persisted.

The text tokenizer applies Unicode NFKC and case folding, then emits words plus
within-word character bigrams and trigrams. Japanese matching therefore does not
depend on whitespace. Vietnamese diacritics are preserved. There is no learned
vocabulary, stopword selection, query expansion or label-derived tuning.

BM25 uses k1=1.2, b=0.75 and positive log-IDF
`log(1 + (N - df + 0.5) / (df + 0.5))`. Statistics come only from the target PDF.
Query terms are deduplicated and sorted so summation is stable across processes.
Ties use ascending physical page number. Blank/error pages stay in the candidate
set with zero lexical signal. Zero-signal queries are counted explicitly.
This is a diagnostic baseline, not a visual or trained semantic retriever.

References: [BM25 scoring](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)
and [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).

## Metrics and interpretation

Budgets 1, 2, 3, 5 and 10 were specified before the live run. A short PDF contributes
all its pages when k exceeds its length. Both methods rank the same full candidate
set. All 16 questions remain in every denominator.

- Recall@k: fraction of gold evidence pages retrieved.
- All-evidence@k: one only when every gold page is present.
- MRR@k: reciprocal rank of the first relevant page, or zero if none occurs.
- MAP@k: mean AP@k, with AP divided by `min(number of gold pages, k)`.
- nDCG@k: binary relevance discounted by rank, normalized to the ideal ranking.

Reports include question averages, equal-document averages, document/language/
answer-format slices, extraction coverage and anonymous per-question scores.
The five PDFs provide a descriptive training diagnostic. No retriever tuning or
cross-validation is claimed. The single Vietnamese example does not estimate
language-level performance. Test labels are unavailable and are not used.

These retrieval measures supplement the [published LAVA metric](https://lava-workshop.github.io/#evaluation).
They cannot substitute for semantic answer credit and predicted evidence-page F1.
The retrieval report stores `local_lava_overall: null` until a reader is evaluated.

## Run and resume

From the project root in Studio:

```bash
make retrieval-preview
make retrieval-evaluate
```

Preview is offline. Evaluate uses the current CPU host, pinned S3 source versions
and SHA-256 checksums. It never launches a GPU. Run the same command to resume.
Native extraction and each question ranking have separate durable checkpoints;
completed summaries are reconstructed from verified checkpoints and checked against
the saved aggregate. Raw questions, PDF text and rankings remain private.

Checkpoint keys bind the implementation, dependency lock, settings and source
hashes. Conditional S3 writes create a pending slot and atomically complete it.
This works with the existing GetObject/PutObject permission grant and requires no
new ListBucket permission. The store never treats access denial as a cache miss,
and rejects corrupted or conflicting completed objects. Source caches are restored
from pinned versions if their local copies disappear or fail verification.

The CPU process ends if Studio stops. S3 checkpoints survive; restarting the same
command restores completed work. Longer GPU jobs use SageMaker's independent job
lifecycle. The [executed notebook](../reports/notebooks/04_evidence_retrieval.ipynb)
can be viewed without running either workload.

## Next experiment

Evaluate the provisional 9B reader with a declared retrieved-page budget, keeping
the model revision, decoding and semantic judge unchanged. Measure the resulting
LAVA answer/evidence/overall scores against the oracle run, including all failures.
Document page selection and full runtime. Use failure analysis to decide whether
multilingual embeddings, visual retrieval or reranking justify their added cost.
Do not select a method from hidden test feedback.

The subsequent milestone is complete test inference and organizer-hardware runtime
verification. Submission-schema validation exists; an accepted Kaggle submission
and authenticated late-submission eligibility remain outstanding.
