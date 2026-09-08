# Portfolio closeout — September 8, 2026

This release completes a component-level document-AI research benchmark. The
employer reading path is **Notebook 00 → 03 → 04**, with 01 and 02 providing
methodology and execution details. Notebook 05 preserves the optional integrated
workflow and clearly marks its answer score as unmeasured.

## Delivered evidence

- Three completed 16-question oracle reader benchmarks: local LAVA scores of
  73.82% (4B), 87.02% (9B), and 80.36% (27B NF4). Invalid answers remain failures.
- Full-document BM25 retrieval over 74 pages: 95.31% recall@5 and 14/16 questions
  with complete evidence; an independent resumed process reused every extraction
  and ranking.
- 1,582 lexical/fusion/exploration configurations, with 349 duplicate-ranking
  rejections. Document-isolated selection retained the baseline; none was promoted.
- A completed visual challenger: visual-only recall@5 was 57.81%; a conservative
  hybrid reached 98.44% and 15/16 complete questions. The hybrid is exploratory and
  has not been independently validated or promoted.
- Six canonical notebooks with a consistent reading path, saved results, tables,
  interpretable figures, and visible execution logs. No renamed repair variants.

These are development findings on 16 questions from five PDFs. The score uses the
published LAVA formula with a pinned local judge; exact organizer parity is not
established. This release makes no held-out, leaderboard, or state-of-the-art claim.

## Cloud verification

The [dated verification record](../reports/portfolio/closeout.json) records the
checks performed against the connected AWS account:

- S3 versioning enabled; all 208 raw data objects present.
- Three public reader summaries matched the repository byte-for-byte by SHA-256.
- All three private result sets passed SHA-256 and unique 16-question coverage
  checks. The 9B and 27B prefixes also contained 16 verified question checkpoints
  each; the earlier 4B prefix contains saved results without per-question checkpoints.
- The visual experiment's original aggregate, internal summary hash, source script
  hash, model revision, and completed job were verified.
- The optional integrated job was stopped during capacity wait. Its terminal
  service record reports one billable second and no verified inference result.
- No active LAVA training jobs or LAVA-named endpoints remained in us-west-2 at
  verification. Stored data and completed experiment artifacts were preserved.

This is a project closeout check, not an account-wide billing audit. It did not
repeat GPU inference or download all raw PDFs again.

## Publication and quality

PR #15 previously failed because its analysis inputs changed without refreshing
the six published notebooks. The stale-output checks were correct and remain in
place. Closeout refreshes the canonical publications from the new inputs, then
requires the full quality gate and verified notebook reuse before merge.

The local execution environment prohibits kernel IPC sockets, so real-kernel
execution and its eight integration tests run in GitHub's Linux environment.
Local tests alone are not the release acceptance gate.

## Final scope

No additional modeling is required for this portfolio release. End-to-end answer
evaluation, application deployment, full test inference, and Kaggle upload remain
optional extensions. Their code and cost controls are retained for intentional
future use; they are not hidden prerequisites for calling this benchmark complete.
