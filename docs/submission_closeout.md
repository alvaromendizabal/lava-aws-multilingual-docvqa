# Submission results and completion status

The strongest recorded submission scored **0.49 public / 0.53 private**. A later
diagnostic scored **0.48 public / 0.49 private**. The October 6, 2026 reconciliation
refreshed both completed submission receipts through an owner-executed, read-only
run. It preserved the better historical artifact and made **zero new submissions**.

| Result | Public score | Private score | Completion caveat |
|---|---:|---:|---|
| Historical best | 0.49 | 0.53 | 624 rows, including two template-derived fallbacks |
| Later diagnostic | 0.48 | 0.49 | 624 rows, including two compatibility abstentions |
| Diagnostic minus best | −0.01 | −0.04 | Same split compared with the same split |

These are competition scores. The **82.68% local result** elsewhere in this
repository belongs to the original frozen development evaluation on **16 questions
across five documents**. It is neither a competition score nor a measurement of the
later recovered prediction composite. Private scores were recorded for reporting,
not used for tuning.

## What is complete

The closeout completed **14 of 14 reconciliation tasks** in 21.88 seconds. It
verified candidate identities, compared the retained predictions, refreshed score
receipts, preserved **1,081 inference checkpoints**, and completed its reporting and
backup checks. The recorded runtime compute estimate was **$0.010052**, excluding
final packaging overhead, idle time, storage, and unallocated billing.

The owner run executed a real Jupyter kernel. Its two code cells completed; Plotly
and offline HTML outputs were saved, reopened, and followed by further execution.
These checks verify execution and persisted output content. They do not assert a
browser rendering test.

## What remains unresolved

The current set contains **624 records**, with **622 passing structural and
non-abstention checks** and **two unresolved answers**. Structural acceptance does
not measure factual correctness. Neither historical artifact passes the project's
strict requirement for 624 supported, non-abstaining answers.

The comparison found 389 changed rows: 376 changed answer strings and 111 changed
evidence-page sets. Inclusion–exclusion gives 98 rows with both changes, 278 with
answer changes only, and 13 with evidence changes only. These are differences in
saved outputs, not evidence of improved accuracy. The later official scores were
lower, so the historical best remained the preferred recorded submission.

## Reproduce the public checks

From the repository root, run the standard-library verifier. It requires no GPU,
cloud account, model download, competition data, or third-party Python packages.

```bash
python examples/verify_submission_closeout.py
```

The verifier reads [the sanitized aggregate evidence](../research/submission_closeout.json)
and recomputes score deltas, structural coverage, overlapping change counts, and
the 14-task accounting. Its output explicitly retains `strict_complete: false` and
`private_inference_reproduced: false`. Regression tests reject false completeness
claims, metric-scope substitutions, inconsistent counts, and nonfinite scores.

The archive digest identifies privately retained evidence; it does not independently
authenticate that evidence. Raw receipts, test predictions, question-level records,
exact private selection rules, cloud locations, and operator orchestration remain
private. Consequently, this public check reproduces aggregate arithmetic and scope
validation, **not full competition inference or the official score**.

The engineering lesson is straightforward: changed predictions and successful
execution are insufficient for promotion. A candidate needs a comparable evaluation,
an intact evidence trail, and a clear completion contract. This project preserves
the stronger measured artifact while reporting the remaining limitations explicitly.

For the broader public workflow, see [reproducibility](reproducibility.md).
