# LAVA Evaluation Protocol

Protocol lock: `23cf9a605fcdc553c2948b1fd1001d8e300187f35a43f46083a3396cc7ac2b61`

## Design

- Five outer leave-one-document-out folds estimate generalization.
- Four inner document folds inside each outer training set govern any label-based tuning.
- No document or question crosses an outer or inner partition.
- Question-micro and document-macro scores are both mandatory.
- All five outer document scores are displayed.
- Architecture development uses external public data first.
- LAVA test or leaderboard feedback cannot become a tuning loop.

## Published challenge metric

Answer correctness and exact evidence-page grounding receive equal weight. String and 
number answers use a Gemma-3 1B semantic judge; unordered lists use optimal one-to-one 
semantic matching followed by F1; ordered lists use semantic LCS; evidence pages use 
exact set F1.

## Retrieval diagnostics

Recall@k, all-evidence success@k, MRR@k, MAP@k, and nDCG@k are frozen at page 
budgets 1, 2, 3, 5, and 10. These diagnose retrieval independently of reader quality.

## Public outer-fold summary

- `outer-01`: validate `doc-01` (4 questions); train on 4 documents / 12 questions; 4 inner folds.
- `outer-02`: validate `doc-02` (4 questions); train on 4 documents / 12 questions; 4 inner folds.
- `outer-03`: validate `doc-03` (3 questions); train on 4 documents / 13 questions; 4 inner folds.
- `outer-04`: validate `doc-04` (4 questions); train on 4 documents / 12 questions; 4 inner folds.
- `outer-05`: validate `doc-05` (1 questions); train on 4 documents / 15 questions; 4 inner folds.

## Evaluator boundary

The organizers publish Gemma-3 1B and the metric formulas but not the exact 
judge prompt, decoding configuration, or checkpoint revision. This local evaluator 
is therefore official-structure-compatible, not claimed server-identical. The 
oracle-reader phase will pin a Gemma runtime and quantify judge sensitivity.
