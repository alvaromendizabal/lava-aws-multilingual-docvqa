# Verified milestones and remaining work — September 10, 2026

This document records completed research milestones, not completion of the full
competition project. The employer reading path is **Notebook 00 → 05 → 03 → 04**;
01 and 02 provide experimental design and cloud execution details.

As of September 10, 2026, 622 of 624 test answers are structurally complete and
preserved. The targeted run recovered all three routed questions at 04:36 UTC;
two suspected source-document mismatches remain. The complete CSV, validated
Kaggle upload and broader feature-research acceptance gate remain unfinished.

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

The final Studio gate completed at **2026-09-08 04:48:23 UTC**: **546 tests passed, with no skips or warnings**, in 41.28 seconds; the complete gate took 61 seconds. Ruff checked 149 files and mypy checked 84 source files. All six notebooks executed end to end, and the integration suite independently exercised their real kernels and source preservation. Published outputs contain no errors or stderr.

| Canonical notebook | Executed code cells |
| --- | ---: |
| 00 — Research overview | 3 |
| 01 — Experiment design | 2 |
| 02 — Cloud execution | 3 |
| 03 — Model quality and cost | 8 |
| 04 — Evidence retrieval | 10 |
| 05 — Complete system | 9 |

The [executed publication commit](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/commit/8008bef2f902b92302c736450a9775f311b1fc9b) contains the six notebooks and six manifests. Their hashes match the independently read-back-verified S3 publication; the successful Studio log is archived with them. The publication manifest SHA-256 is `fd380f347a001b7236f953b35121325d0de7a382023c466267c59695e88d7e0b`.

The generated HTML report was also opened in Studio: the static fallback rendered, Plotly loaded, and legend selection changed the displayed series. Static figures were rendered and visually inspected; embedded JavaScript passed syntax checking.

[PR #16](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/pull/16) contains the implementation and reviewed outputs; its [checks](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/pull/16/checks) provide the independent Linux CI record. [GitHub validation run 34188449609](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/runs/34188449609) passed on the published notebook commit. Local checks alone are not the release gate. No stale-output or kernel test was disabled to publish the result.

## Completed milestones and remaining acceptance criteria

The completed work can be reviewed as an applied ML engineering portfolio. It
demonstrates measured system behavior, retrieval research, rejected complexity,
a targeted improvement, durable execution and reproducible public evidence.

All quality findings use 16 previously examined training questions from five PDFs: 15 Japanese questions and one Vietnamese question. The metric follows the published LAVA formula with a pinned local judge; exact organizer prompt/runtime parity is not established. No held-out, language-wide, leaderboard or state-of-the-art result is claimed.

The finite lexical grid is complete. It does not establish exhaustive document
feature research. A new 17-policy domain-feature audit measured body, heading,
block, table, numeric, query-coverage and adjacent-page signals. Block retrieval
improved pooled complete-evidence coverage from 14/16 to 15/16, concentrated in one
training document; all five conservative document folds retained the baseline.
The complete feature-family ablations and coverage limitations are in Notebook 04
and [the research acceptance gate](retrieval.md#feature-research-completion-gate).

Completion still requires justified disposition of the remaining feature families,
appropriate validation for any promoted policy, resolution of the unanswered test
questions, a canonically validated 624-row CSV and a verified Kaggle receipt. The
existing development scores must remain separate from any eventual competition
score. Application hosting is not required.
