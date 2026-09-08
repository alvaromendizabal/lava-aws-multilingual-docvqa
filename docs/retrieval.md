# Full-document evidence retrieval

Notebook 04 asks whether the system can find the pages needed to answer a question.
The earlier reader comparison received gold evidence pages. Its 9B score remains
an oracle reader measurement until we actually run 9B on retrieved evidence.

## Completed baseline experiment

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

The production text tokenizer applies Unicode NFKC and case folding, then emits
words plus within-word character bigrams and trigrams. Japanese matching therefore
does not depend on whitespace. Vietnamese diacritics are preserved. There is no
learned vocabulary, stopword selection, query expansion or label-derived tuning.

BM25 uses k1=1.2, b=0.75 and positive log-IDF
`log(1 + (N - df + 0.5) / (df + 0.5))`. Statistics come only from the target PDF.
Query terms are deduplicated and sorted so summation is stable across processes.
Ties use ascending physical page number. Blank/error pages stay in the candidate
set with zero lexical signal. Zero-signal queries are counted explicitly.

## Exhaustive lexical feature research

The production baseline is intentionally simple, but the feature search is not.
The research-only implementation in `src/lava/retrieval/feature_research.py`
generates **1,089 scalar query-page BM25 features** across 11 multilingual lexical
views, 11 k1 values and 9 length-normalization values. The views include words,
within-token character n-grams, mixed word/character signals, and whitespace-free
character n-grams that can bridge tokenization boundaries.

Every complete ranking is generated before gold evidence is inspected. Ranking
signatures are then deduplicated without labels: 981 of the 1,089 BM25 candidates
were unique and 108 were rejected as redundant. A second search generated 493
conservative RRF, Borda and baseline-preserving exploration policies; 252 produced
unique rankings and 241 were rejected as duplicates. The full search therefore
examined 1,582 candidate configurations while keeping selection separate from the
hidden test set.

Pooled diagnostics show why naive feature-count optimization is dangerous. A
whitespace-free character-trigram BM25 variant retrieved every labeled evidence
page within k=5 on all 16 questions, and character 2/3/4-gram views also improved
pooled recall. Word-only retrieval was substantially worse, consistent with the
multilingual document setting. Those results are descriptive, not selection proof.

For selection, each outer fold holds out one complete document. Free selection
among the broad BM25 grid dropped to 90.77% oracle grounding-F1@5, 81.25%
complete-evidence@5 and 89.06% recall@5, below the fixed baseline. Candidate choices
also changed across folds. In a separate conservative fusion search where the
existing baseline remained an eligible no-change option, the baseline was selected
on all five outer folds and reproduced its 97.02% oracle grounding-F1@5, 87.50%
complete-evidence@5 and 95.31% recall@5.

The decision is therefore to **retain the production BM25 baseline**. A perfect
pooled result is rejected because it does not survive document-isolated validation.
This is intentional feature selection: broad generation, label-blind redundancy
removal, grouped holdout testing, and removal of complexity without robust gain.
The exact aggregate evidence is checksum-bound in
[`reports/retrieval/feature_search.json`](../reports/retrieval/feature_search.json).

The next justified challenger is visual document retrieval, not more lexical
hyperparameter search. ColPali/ColQwen-style late-interaction page-image retrieval
is relevant because LAVA explicitly requires visual understanding of tables,
figures, photographs and layout that native PDF text can miss. It will be compared
as a challenger and will not replace BM25 without held-out evidence.

## Metrics and interpretation

Budgets 1, 2, 3, 5 and 10 were specified before the original baseline run. A short
PDF contributes all its pages when k exceeds its length. All methods rank the same
full candidate set, and all 16 questions remain in every denominator.

- Recall@k: fraction of gold evidence pages retrieved.
- All-evidence@k: one only when every gold page is present.
- MRR@k: reciprocal rank of the first relevant page, or zero if none occurs.
- MAP@k: mean AP@k, with AP divided by `min(number of gold pages, k)`.
- nDCG@k: binary relevance discounted by rank, normalized to the ideal ranking.
- Oracle grounding-F1@k: the best evidence F1 possible if a downstream reader could
  perfectly keep only the gold pages that appeared in the retrieved candidate set.

Reports include question averages, equal-document averages, document/language/
answer-format slices, extraction coverage and anonymous per-question scores.
The five PDFs remain a small training diagnostic. The feature search adds
leave-one-document-out selection, but the single Vietnamese example still cannot
estimate language-level generalization. Hidden test labels and leaderboard feedback
are not used.

These retrieval measures supplement the published LAVA metric. They cannot
substitute for semantic answer credit and predicted evidence-page F1. The main
retrieval report stores `local_lava_overall: null` until a reader is evaluated on
retrieved pages.

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
lifecycle. The executed Notebook 04 can be viewed without running either workload.

## Research boundary

The lexical feature search is complete and did not justify changing production
retrieval. The integrated 9B retrieved-evidence pilot is a separate measured system
experiment. A visual-retrieval benchmark and complete 624-question test inference
remain separate work until they have real measured outputs. No accepted Kaggle
submission or leaderboard score is claimed by this repository.
