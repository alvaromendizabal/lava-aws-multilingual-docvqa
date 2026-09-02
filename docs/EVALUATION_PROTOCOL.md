# LAVA evaluation protocol

The labeled LAVA set contains only 16 questions across five source PDFs. The project therefore uses **nested leave-one-document-out evaluation**, not a random question split.

## Nested validation

Five outer folds estimate document-level generalization. In every outer fold, one complete PDF is held out and all questions from that PDF remain together. The remaining four PDFs form the outer training set.

Any tuning that uses LAVA labels must occur only inside four inner leave-one-document-out folds built from the corresponding outer training set. The outer validation PDF is never visible to an inner fold. Architecture and hyperparameter development should rely primarily on external public multilingual Document VQA data because the LAVA labels are too scarce to support broad search.

Question-micro mean is the primary development score because it matches the challenge's final mean across questions. Document-macro mean and all five outer document scores are mandatory secondary results. A single aggregate may not hide a severe document failure.

## Published metric structure

The evaluator implements the challenge's public formulas:

- string and number answers use binary semantic equivalence;
- unordered lists use **maximum-cardinality one-to-one semantic matching** followed by F1;
- ordered lists use semantic longest common subsequence divided by the larger list length;
- evidence pages use exact set F1;
- each question's overall score is the equal mean of answer and grounding scores;
- the final score is the mean across questions.

Greedy list matching is prohibited because its score can depend on item order. The implementation uses Hopcroft-Karp maximum bipartite matching.

## Retrieval diagnostics

Before reader evaluation, evidence retrieval is measured at page budgets 1, 2, 3, 5, and 10 using:

- Recall@k;
- all-evidence success@k;
- reciprocal rank@k;
- average precision@k;
- nDCG@k.

These metrics isolate page retrieval from answer generation. A fluent reader cannot conceal a weak retrieval system.

## Semantic-judge boundary

The challenge page identifies Gemma-3 1B as the semantic judge and publishes the metric formulas. It does not publish the exact judge prompt, decoding implementation, or checkpoint revision. This repository therefore claims **official-structure compatibility**, not bitwise organizer parity. Before any model result is reported, the downloaded Gemma checkpoint commit, tokenizer revision, prompt hash, decoding configuration, container digest, and judge cache must be frozen.

## Small-sample inference

The five source PDFs are the independent evaluation clusters. Every comparison must show raw per-document effects. Document-bootstrap intervals are descriptive. Challenger-versus-incumbent comparisons use an exact two-sided paired sign-flip test across document effects. Question-level tests that pretend 16 questions are independent are prohibited.

## Promotion gate

A challenger is promoted only when it:

1. improves the prespecified question-micro score or produces a justified Pareto improvement;
2. improves at least three of five outer documents;
3. reports answer and grounding changes separately;
4. reports every language, answer-format, evidence-cardinality, and document-complexity slice;
5. records latency, throughput, peak GPU memory, token use, and estimated cloud cost;
6. remains within the open-model, one-A100, 40 GB, two-hour inference constraint;
7. is reproducible from source, data, split, model, prompt, container, and random-seed fingerprints.

## Test-set policy

The 624 unlabeled test questions are never used for architecture selection, prompt selection, retrieval-weight tuning, threshold selection, or error-driven iteration. Test predictions are generated only from a frozen candidate. Any late-leaderboard score is an external final measurement, not a development fold.
