/** Evaluate the evaluator: schema failures, label isolation, metrics and source drift. */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { ROOT, SUITE, REPORT, PAGE, evaluate, encode, render, readSource, validateSuite, verifyReport, main } from "./evaluate_public_challenges.mjs";
import { sources, loadEngine, runCase, summarize } from "./public_challenge_adapter.mjs";

const engine = await loadEngine(name => readSource(ROOT, name).toString("utf8"));
const suite = JSON.parse(readSource(ROOT, SUITE));
const original = await evaluate();
const clone = value => structuredClone(value);
let count = 0;
async function test(name, callback) {
  await callback();
  console.log(`PASS ${++count} ${name}`);
}
function mutated(change) { const copy = clone(suite); change(copy); return copy; }

await test("real engine evaluation exactly reproduces the tracked report", () => verifyReport(original, readSource(ROOT, REPORT)));
await test("all authored cases are present once and failures remain visible", () => {
  assert.equal(original.cases.length, suite.cases.length);
  assert.deepEqual(original.cases.map(row => row.id), suite.cases.map(row => row.id));
  assert.ok(original.summary.failures > 0);
  assert.equal(original.summary.correct + original.summary.failures, original.summary.cases);
  assert.equal(Object.values(original.by_slice).reduce((sum, row) => sum + row.cases, 0), suite.cases.length);
  assert.equal(Object.values(original.by_group).reduce((sum, row) => sum + row.cases, 0), suite.cases.length);
});
await test("repeated evaluation is byte deterministic", async () => assert.equal(encode(original), encode(await evaluate())));
await test("HTML report is current, accessible without JavaScript and escapes query text", () => {
  assert.equal(render(original), readSource(ROOT, PAGE).toString());
  const changed = clone(original); changed.cases[0].query = '<script>alert("unsafe")</script>';
  assert.ok(render(changed).includes("&lt;script&gt;"));
  assert.ok(!render(changed).includes("<script>"));
  assert.match(render(original), /scope="col"/);
  assert.match(render(original), /Skip to case results/);
});
await test("changing expected labels cannot change engine decisions", () => {
  for (const row of suite.cases) {
    const changed = clone(row);
    if (Object.hasOwn(changed, "expected")) changed.expected = changed.expected === "allowed" ? "violation" : "allowed";
    else changed.expected_line = changed.expected_line === null ? engine.documents.find(doc => doc.id === changed.document).pages[0].lines[0].id : null;
    const left = runCase(row, engine), right = runCase(changed, engine);
    const inferenceKeys = ["decision", "state", "margin", "coverage", "actual_line", "actual_page", "status", "lexical_coverage"];
    for (const key of inferenceKeys) assert.equal(left[key], right[key]);
  }
});
await test("empty metric denominators are null, never NaN or optimistic perfect scores", () => {
  const empty = summarize([]);
  assert.equal(empty.cases, 0);
  assert.equal(empty.agreement, null);
  assert.ok(!JSON.stringify(empty).includes("NaN"));
});
await test("case-weighted agreement is not an unweighted average of slice rates", () => {
  const sample = [original.cases[0], original.cases[0], original.cases.find(row => !row.correct)];
  assert.equal(summarize(sample).agreement, 0.666666666667);
});
await test("missing and duplicate case identities fail", () => {
  assert.throws(() => validateSuite(mutated(x => delete x.cases[0].id), engine), /case ID/);
  assert.throws(() => validateSuite(mutated(x => x.cases.push(x.cases[0])), engine), /case ID/);
});
await test("unknown expected labels or passages fail", () => {
  assert.throws(() => validateSuite(mutated(x => {
    if (Object.hasOwn(x.cases[0], "expected")) x.cases[0].expected = "unknown";
    else x.cases[0].expected_line = "missing-passage";
  }), engine), /expected|Expected/);
});
await test("unknown context and extra case fields fail", () => {
  assert.throws(() => validateSuite(mutated(x => {
    const key = Object.hasOwn(x.cases[0], "policy") ? "policy" : "document";
    x.cases[0][key] = "missing";
  }), engine), /Unknown/);
  assert.throws(() => validateSuite(mutated(x => x.cases[0].hidden_label = 1), engine), /fields/);
});
await test("empty, non-string and oversized queries fail", () => {
  for (const query of [" ", 1, "x".repeat(1201)])
    assert.throws(() => validateSuite(mutated(x => x.cases[0].query = query), engine), /query/);
});
await test("invalid suite identity, scope and counts fail", () => {
  for (const change of [x => x.schema = 2, x => x.scope = "MODEL_BENCHMARK", x => x.cases = [], x => x.cases = null, x => x.extra = true])
    assert.throws(() => validateSuite(mutated(change), engine));
});
await test("changed row outcomes and altered source fingerprints fail verification", () => {
  const altered = clone(original); altered.cases[0].correct = !altered.cases[0].correct;
  assert.throws(() => verifyReport(altered, readSource(ROOT, REPORT)), /stale/);
  const changed = clone(original); changed.sources[sources[0]] = "0".repeat(64);
  assert.throws(() => verifyReport(changed, readSource(ROOT, REPORT)), /stale/);
});
await test("unsafe source paths are rejected", () => {
  for (const name of ["../README.md", "/etc/passwd", "public-demo/../README.md", "public-demo\\engine.js"])
    assert.throws(() => readSource(ROOT, name), /Unsafe/);
});
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "public-challenges-"));
try {
  for (const name of [...Object.keys(original.sources), REPORT, PAGE]) {
    const destination = path.join(temp, name);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, readSource(ROOT, name));
  }
  await test("a source-only change invalidates a prior result even when predictions match", async () => {
    const name = sources.find(name => name.endsWith("engine.js"));
    fs.appendFileSync(path.join(temp, name), "\n// provenance changed\n");
    const changed = await evaluate(temp);
    assert.deepEqual(changed.cases, original.cases);
    assert.throws(() => verifyReport(changed, readSource(temp, REPORT)), /stale/);
    fs.writeFileSync(path.join(temp, name), readSource(ROOT, name));
  });
  await test("symlinked sources fail instead of reading beyond the checkout", async () => {
    const file = path.join(temp, sources[0]);
    fs.unlinkSync(file); fs.symlinkSync(path.join(ROOT, sources[0]), file);
    await assert.rejects(evaluate(temp), /Symlink/);
    fs.unlinkSync(file); fs.writeFileSync(file, readSource(ROOT, sources[0]));
  });
  await test("explicit report updates preserve deterministic bytes and do not leave staging files", async () => {
    await main(["--write"], temp);
    assert.equal(readSource(temp, REPORT).toString(), encode(original));
    assert.deepEqual(fs.readdirSync(path.join(temp, "reports")).sort(), ["public_challenges.html", "public_challenges.json"]);
  });
  await test("dangling report symlinks cannot redirect publication", async () => {
    fs.unlinkSync(path.join(temp, REPORT));
    const outside = path.join(temp, "must-not-write.json");
    fs.symlinkSync(outside, path.join(temp, REPORT));
    await assert.rejects(main(["--write"], temp), /Symlink/);
    assert.equal(fs.existsSync(outside), false);
  });
} finally { fs.rmSync(temp, { recursive: true, force: true }); }
await test("CLI fails closed on unknown flags", () => {
  const result = spawnSync(process.execPath, [path.join(ROOT, "tools/evaluate_public_challenges.mjs"), "--pretend-pass"], { encoding: "utf8" });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Usage:/);
});
console.log(`${count} challenge-evaluation tests passed`);
