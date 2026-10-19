# Document intelligence: verified public review

This review checks lexical retrieval, structured citations, answer-export replay and the recorded evaluation boundaries.
[Project and results](../README.md) · [Inspection brief](index.html) · [Exact check configuration](project.json)

## One command, an inspectable result

From the repository root, with Python 3.11+ and Node.js:

```bash
python3 -B -S review/run.py --output /tmp/project-review --with-node
python3 -B -S review/run.py --output /tmp/project-review --with-node --check
```

Use a writable directory outside the checkout; on Windows replace `/tmp/project-review`
with an appropriate local path. Open the generated `index.html`. The bundle includes
actual command logs, `receipt.json`, `manifest.json`, and the configured diagnostic
outputs. No AWS account, private dataset or private weights are required.

Omit `--with-node` for Python-only checks. Omitted browser tests are never reported
as passed. The [configuration](project.json) lists the exact commands, source files,
evidence and browser assets covered; it is not a claim of whole-system private replay.

## Verify rather than silently overwrite

An unchanged rerun checks the saved bundle before reuse. `--check` only verifies.
Changed source, damaged artifacts, mismatched modes and failed commands cause a
nonzero exit. After an intentional source change, use a **new output directory**.
The default per-command deadline is 90 seconds; `--timeout` changes it explicitly.

The receipt records runtime versions, UTC times, elapsed checks and inspected hashes.
Checksums establish consistency, not independent authentication of historical model
scores. The committed inspection page is a guide, **not an execution receipt**.
Public examples and aggregate replay are distinct from training or running the
withheld competitive model; see the project README for that boundary.

[Evidence-page retrieval and its measured scope](../docs/retrieval.md)

## Maintain this review

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -q
python3 -B -S review/run.py --check-page
```

After changing the configuration or template, regenerate the inspection page with
`python3 -B -S review/run.py --render`, inspect it, and rerun both checks. Full package
training and notebook requirements remain in the project's main reproduction guide.
