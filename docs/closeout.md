# Verified project closeout — September 8, 2026

This release completes an evaluated retrieval-to-answer research system. The employer reading path is **Notebook 00 → 05 → 03 → 04**; 01 and 02 provide experimental design and cloud execution details.

## Delivered and measured

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Three oracle readers | Local LAVA: 4B 73.82%, 9B 87.02%, 27B NF4 80.36% | 9B is strongest among the tested configurations; size alone is not isolated |
| Full-PDF BM25 | 95.31% recall@5; all evidence for 14/16 questions | Strong evidence coverage on the labeled pilot |
| Integrated first pass | 67.57% local LAVA; 48.90% answer credit; 86.25% evidence F1 | Component scores did not predict complete-system behavior |
| Citation-guided reread | 77.99% local LAVA; 67.65% answer credit; 88.33% evidence F1 | +10.42 points with the same 9B reader; post-hoc development result |
| Visual-only retrieval | 57.81% recall@5; 8/16 complete questions | Underperforms the lexical baseline |
| Four lexical pages + one novel visual page | 98.44% recall@5; 15/16 complete questions | Exploratory recovery of one question; not promoted |

Both integrated passes produce 16 valid responses. The reread improves three document scores and ties two. Equal-document LAVA improves from 67.00% to 76.16%; the exact two-sided sign-flip p-value is 0.25. The extra pass reduces input pages from 76 to 27 but also reduces complete input-evidence coverage from 14/16 to 12/16. Both passes and all failures remain published.

The two completed GPU jobs used one L40S each, with 409 and 355 billable seconds. At the dated regional Training rate their combined estimate is $1.2012. Studio, storage, other attempts and other charges are excluded.

## Feature generation, screening and selection

The completed source-bound audit evaluates all **1,582 configurations** in the declared search space: 1,089 BM25 configurations across 11 text representations and 99 parameter pairs, plus 493 fusion/exploration policies.

| Accounting | Generated | Unique within stage | Duplicate within stage |
| --- | ---: | ---: | ---: |
| BM25 | 1,089 | 983 | 106 |
| Fusion/exploration | 493 | 408 | 85 |

One additional signature overlaps between stages: **1,390 unique signatures and 192 global duplicates**. These global counts describe the search. Each fold independently screens and selects using only its four training documents; all ten fold decisions and screening counts are public.

Whitespace-free trigrams reach perfect pooled evidence recall, while word-only retrieval is substantially weaker. Those pooled family comparisons do not survive as a reliable selection gain: free document-isolated BM25 selection reduces recall@5 to 89.06%. The conservative multi-document gate retains the original BM25 in all five folds. No new lexical configuration is retained as the default.

All 12 feature families were executed on the existing CPU host in 10.973 seconds. An independent process reused 12/12 checkpoints in 1.783 seconds and reproduced the same summary checksum. The audit supersedes the earlier aggregate with fully specified fusion views and executable selection rules.

## Implementation completed

- Added scalable cached BM25 evaluation, immutable feature-family checkpoints, deterministic ranking signatures, training-only selection, full fold audits and recovery tests.
- Completed the real full-PDF retrieval → 9B answer → citation validation → semantic evaluation path.
- Implemented and ran a frozen reread policy using only first-pass citations; validated the complete input transformation against saved cloud artifacts.
- Added a bounded alternate-host retry for unavailable GPU capacity, preserving deterministic jobs, per-question checkpoints and the original inference contract.
- Published static Matplotlib figures and a Plotly report with static fallbacks, bound to verified source and result hashes.
- Updated all six canonical notebooks, the README, architecture, retrieval methods and operator documentation to reflect measured complete-system behavior.

The fixed semantic judge, its 28 development controls, model revisions, original reader results and labeled-data boundary are preserved.

## Cloud artifacts verified

The [dated machine-readable closeout](../reports/portfolio/closeout.json) records scope and evidence:

- S3 versioning is enabled and the original audit confirmed all 208 raw files.
- Three original reader result sets each passed checksum and unique 16-question coverage checks. The 9B and 27B runs also have 16 verified question checkpoints each; the earlier 4B run stores complete results without per-question checkpoints.
- The visual output, internal checksum, model revision, original source hash and completed AWS job were verified.
- Both integrated inference objects and all **32 answer checkpoints** matched their saved records by SHA-256 and were independently parsed.
- The 16 refinement inputs exactly match the frozen transformation of the first-pass predictions; page selection uses no reference labels.
- The lexical audit summary and both independent execution receipts match immutable cloud artifacts.
- All LAVA training jobs are terminal and no LAVA-named endpoint is active in us-west-2. The existing Studio workspace and stored artifacts remain available.

The original reader/visual verification and the later integrated verification have separate timestamps. This is a project closeout, not an account-wide billing audit.

## Publication and quality

The required gate covers the frozen environment, Ruff formatting/lint, mypy, all pipeline shell syntax, pytest, compilation, notebook hygiene and Git whitespace. Public notebooks must have complete sequential execution, no error or stderr outputs, current source/input/output hashes and safe default controls.

The real-kernel run and final GitHub CI result are recorded with the release publication. Local execution alone is not sufficient: this scratch environment prohibits kernel IPC, so actual notebook and kernel integration verification use Linux Studio and GitHub runners.

## Completed scope

The project is ready to review as an applied ML engineering portfolio. It demonstrates measured system behavior, broad retrieval research, rejected complexity, a targeted improvement, durable execution and reproducible public evidence.

All quality findings use 16 previously examined training questions from five PDFs: 15 Japanese questions and one Vietnamese question. The metric follows the published LAVA formula with a pinned local judge; exact organizer prompt/runtime parity is not established. No held-out, language-wide, leaderboard or state-of-the-art result is claimed.

Application hosting, 624-question test inference and Kaggle upload are outside this measured release. The optional export code remains available, but these operations are not unfinished acceptance criteria for the research system.
