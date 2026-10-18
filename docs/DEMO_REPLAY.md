# Replay a Document Desk export

An exported answer is useful only when another person can check how it was produced. The replay command executes the same retrieval engine and export builder as the browser, using the public authored documents. It verifies the original result and independently revalidates any human-edited response.

## Run and verify

Use Node.js 24.19.0, the CI runtime. No packages, cloud account, model download or document upload is needed.

```bash
node tools/replay_public_demo.mjs --output /tmp/document-replay
node tools/replay_public_demo.mjs --output /tmp/document-replay --check
```

The first command creates a reference export and verifies it. The second recomputes that export and checks the stored receipt. To use the JSON downloaded from Document Desk, choose a new output directory:

```bash
node tools/replay_public_demo.mjs --input lava-evidence-aoba-operations.json --output /tmp/my-document-replay
```

On Windows, replace `/tmp/...` with an output directory whose parent exists. An existing matching bundle is reused without rewriting its bytes. An unrelated directory is rejected.

| File | Purpose |
|---|---|
| `export.json` | Original retrieval, selected physical pages, source passages, and separate reviewed response |
| `receipt.json` | Result of replay, Node version, UTC timestamp and SHA-256 identities of the actual computation sources |
| `manifest.json` | Checksums of the two bundle files |

## What replay establishes

The query and retrieval settings are rerun against the repository's authored document. Every original answer, citation, page score, support measure, source passage and provenance field must match. A reviewed response is treated as a separate human input. Its citation bounds, exact passage support, abstention consistency and modification flag are recomputed.

Only BM25 page and passage scores allow a numeric tolerance of `1e-12 × (1 + abs(expected))` to accommodate last-bit math-library differences between browser and Node runtimes; zero remains exact. Answers, page order, citations, confidence, settings, edit flags and all other fields must match exactly. Material score changes fail verification.

Changing an export and recalculating its manifest does not conceal a changed generated answer: the computation is run again. Changed source or Node versions require creating a new bundle under the selected runtime. The tests also cover both languages, non-default settings, unsupported questions, confidence edits, abstentions, source drift, oversized inputs, symlinks and preservation of unrelated output directories.

```bash
node tools/test_replay_public_demo.mjs
```

## Interpretation

Replay establishes computational consistency. It does not authenticate who produced an export and it does not establish that an extracted passage answers the question. The visible [challenge suite](PUBLIC_CHALLENGES.md) still preserves four unsupported-overlap extractions and one paraphrase abstention. Those failures remain reproducible; this change does not tune the engine to the visible questions or revise any historical research result.

[Implementation](../tools/replay_public_demo.mjs) · [Shared browser export builder](../public-demo/app.js) · [Failure-path tests](../tools/test_replay_public_demo.mjs)
