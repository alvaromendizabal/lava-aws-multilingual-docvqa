import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { load, replay, referenceExport, createBundle, verifyBundle, main } from "./replay_public_demo.mjs";

const context = load();
const plain = value => JSON.parse(JSON.stringify(value));
const sample = referenceExport(context);
let passed = 0;
function test(name, run) { run(); console.log(`PASS ${++passed} ${name}`); }
function withDirectory(run) {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "lava-replay-test-"));
  try { run(parent, path.join(parent, "bundle")); }
  finally { fs.rmSync(parent, { recursive: true, force: true }); }
}
test("both languages, non-default retrieval settings and abstentions replay", () => {
  for (const document of context.fixture.documents)
    for (const example of document.examples)
      for (const minimumCoverage of [0.2, 0.8]) {
        const result = context.engine.run(document, example.query, { topK: 3, minimumCoverage });
        const payload = plain(context.app.buildExport(result, document, context.fixture.provenance, result.response, context.engine));
        assert.equal(replay(payload, context).status, "verified");
      }
});
test("human confidence edits are preserved separately from generated results", () => {
  const document = context.fixture.documents[0];
  const generated = context.engine.run(document, document.examples[0].query);
  const edited = { ...plain(generated.response), confidence: 0.123 };
  const payload = plain(context.app.buildExport(generated, document, context.fixture.provenance, edited, context.engine));
  assert.equal(replay(payload, context).reviewed_response_modified, true);
  assert.equal(payload.response.confidence, generated.response.confidence);
});
test("human abstention is accepted without claiming semantic verification", () => {
  const document = context.fixture.documents[0];
  const generated = context.engine.run(document, document.examples[0].query);
  const payload = plain(context.app.buildExport(generated, document, context.fixture.provenance, { answer: "", evidence_pages: [], confidence: 0, abstain: true }, context.engine));
  assert.equal(replay(payload, context).semantic_correctness_verified, false);
});
test("unsupported overlapping queries remain reproducible failures, not asserted correct answers", () => {
  const document = context.fixture.documents[0];
  const generated = context.engine.run(document, "東京拠点の売上金額は？");
  const payload = plain(context.app.buildExport(generated, document, context.fixture.provenance, generated.response, context.engine));
  assert.equal(generated.status, "EXTRACTED");
  assert.equal(replay(payload, context).semantic_correctness_verified, false);
});
for (const [label, mutate] of [
  ["generated answer", x => x.response.answer = "invented"],
  ["citation", x => x.response.evidence_pages = [99]],
  ["retrieval score", x => x.retrieval.ranking[0].score += 1],
  ["source passage", x => x.source_document.evidence_pages[0].text += "changed"],
  ["semantic claim", x => x.reviewed_validation.semantic_correctness_verified = true],
  ["review edit flag", x => x.reviewed_response_modified = true],
  ["unsupported edit", x => x.reviewed_response.answer = "not in the document"],
  ["unknown document", x => x.document_id = "private-document"],
  ["extra evidence", x => x.hidden_predictions = [1, 2]],
]) test(`reject altered ${label}`, () => { const changed = plain(sample); mutate(changed); assert.throws(() => replay(changed, context)); });
test("bundle creation, byte-preserving reuse and independent verification", () => withDirectory((parent, output) => {
  const input = path.join(parent, "input.json");
  fs.writeFileSync(input, JSON.stringify(sample));
  assert.equal(createBundle(output, input).reused, false);
  const receipt = fs.readFileSync(path.join(output, "receipt.json"));
  assert.equal(createBundle(output, input).reused, true);
  assert.deepEqual(fs.readFileSync(path.join(output, "receipt.json")), receipt);
  assert.equal(verifyBundle(output).status, "verified");
}));
test("recomputed manifest hashes cannot hide changed inference", () => withDirectory((parent, output) => {
  createBundle(output);
  const file = path.join(output, "export.json");
  const changed = JSON.parse(fs.readFileSync(file)); changed.response.answer += "altered";
  fs.writeFileSync(file, JSON.stringify(changed));
  const manifestPath = path.join(output, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath));
  manifest.files["export.json"] = createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  assert.throws(() => verifyBundle(output));
}));
test("input bounds, symlink protection and unknown output preservation", () => withDirectory((parent, output) => {
  const input = path.join(parent, "input.json"); fs.writeFileSync(input, JSON.stringify(sample));
  fs.symlinkSync(input, path.join(parent, "linked.json"));
  assert.throws(() => createBundle(output, path.join(parent, "linked.json")), /Symlink/);
  fs.writeFileSync(input, "x".repeat(1_000_001));
  assert.throws(() => createBundle(output, input), /1 MB/);
  fs.mkdirSync(output); fs.writeFileSync(path.join(output, "keep.txt"), "keep");
  assert.throws(() => createBundle(output));
  assert.equal(fs.readFileSync(path.join(output, "keep.txt"), "utf8"), "keep");
}));
test("source drift, runtime drift and changed review claims fail independently", () => withDirectory((parent, output) => {
  createBundle(output);
  const receiptPath = path.join(output, "receipt.json");
  const original = JSON.parse(fs.readFileSync(receiptPath));
  for (const mutate of [x => x.runtime = "v0.0.0", x => x.source_sha256["public-demo/engine.js"] = "0".repeat(64), x => x.result.semantic_correctness_verified = true]) {
    const receipt = plain(original); mutate(receipt); fs.writeFileSync(receiptPath, JSON.stringify(receipt));
    const manifestPath = path.join(output, "manifest.json");
    const manifest = JSON.parse(fs.readFileSync(manifestPath));
    manifest.files["receipt.json"] = createHash("sha256").update(fs.readFileSync(receiptPath)).digest("hex");
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    assert.throws(() => verifyBundle(output));
  }
}));
test("CLI fails closed on ambiguous and incomplete arguments", () => {
  for (const args of [["--unknown"], ["--input"], ["--check"], ["--output", "a", "--output", "b"], ["--check", "--output", "a", "--input", "b"]]) assert.throws(() => main(args));
});
console.log(`${passed} document replay tests passed.`);
