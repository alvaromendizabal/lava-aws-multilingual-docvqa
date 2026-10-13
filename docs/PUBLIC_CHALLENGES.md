# Measure retrieval, answering, and abstention separately

**Correct retrieval and a valid citation do not establish a correct answer.** This executable review makes that distinction concrete on 24 visible authored questions over the two fictional demo documents. It runs the shipped BM25/extractive engine with unchanged settings and retains all five diagnostic failures.

```bash
node tools/evaluate_public_challenges.mjs --check
node tools/test_public_challenges.mjs
```

Use Node.js 24.19.0, the CI runtime. No installation, credentials, model download, or cloud compute is required. Open [`review/public_challenges.html`](../review/public_challenges.html) locally for the responsive, printable review. The [JSON report](../review/public_challenges.json) includes every passage/page decision, language and behavior slices, denominators, and source SHA-256 digests. The [authored questions and expected passages](../examples/public_challenges.json) are public.

## Results and failure analysis

| Check | Measured count | Interpretation |
|---|---:|---|
| Expected page ranked first | 18 / 18 answerable questions | Retrieval finds the authored evidence page. |
| Expected passage extracted | 17 / 18 answerable questions | One Japanese paraphrase abstains despite retrieving the right page. |
| Out-of-domain abstention | 2 / 2 | Unrelated questions receive no answer or citation. |
| Unsupported queries with overlapping words | 0 / 4 abstentions | Related vocabulary triggers extraction of a passage that does not answer the question. |
| Expected complete outcome | 19 / 24 | Includes both correct passages and correct abstentions. |
| Schema and extractive-grounding checks | 24 / 24 | Includes abstentions; these checks alone do not establish semantic correctness. |

The four unsupported-overlap cases ask for Tokyo revenue (the document gives orders), Sapporo orders (only three other cities appear), Standard-plan headcount (the document gives fees), and an Enterprise-plan fee (that plan is absent). All four return a copied passage with a valid physical-page citation. Their failure is semantic relevance, not malformed output or invented text. The [per-case report](../review/public_challenges.json) makes each mismatch visible.

The engine returns 21 non-abstaining answers; **17/21** match their authored passage. That denominator differs from the **17/18** answerable-case passage agreement. The per-language report separates Japanese **9/12** from Vietnamese **10/12** expected outcomes. No engine parameter was adjusted to improve these results.

## Evaluation contract

These are deliberately constructed, visible diagnostic cases, including existing UI queries and new variations. They are not a random or untouched benchmark, a language-model evaluation, or an estimate of deployment accuracy. Expected answers identify a complete source passage; this is passage agreement, not the research LAVA metric. No claim about statistical uncertainty or multilingual generalization follows from two fictional documents.

Only a document and question enter `run()`. Expected passage IDs enter the evaluator afterward. Unicode decomposition is applied to one Vietnamese query before inference. Page recall and reciprocal rank use only answerable questions; unsupported queries are assessed for abstention. Rates with zero denominators are `null`. Reports preserve wrong answers even when their schema and copied evidence are valid.

`--check` recomputes both reports and rejects changed outcomes, source fingerprints, or HTML bytes. SHA-256 covers the evaluator, adapter, case suite, casefold table, synthetic documents, and actual engine. The employer review additionally fingerprints the declared public assets, relevant Python parity sources, challenge tests, and reports. This is an explicit review surface, not every transitive repository dependency.

For an intentional revision, inspect the case and source changes, run `node tools/evaluate_public_challenges.mjs --write`, review both report diffs, and run the checks again. Normal invocation prints JSON without writing. CI checks failure paths and retains the reports as artifacts. Historical model results, notebooks, private documents, and active research recipes remain unchanged.
