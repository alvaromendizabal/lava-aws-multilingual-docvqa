/** Replay a Document Desk export against its actual retrieval and validation code. */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { createHash, randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const SOURCES = ["public-demo/unicode-casefold.js", "public-demo/data.js", "public-demo/engine.js", "public-demo/app.js", "tools/replay_public_demo.mjs"];
const LIMIT = 1_000_000;
const plain = value => JSON.parse(JSON.stringify(value));
const encode = value => JSON.stringify(value, null, 2) + "\n";
const digest = value => createHash("sha256").update(value).digest("hex");
function require(condition, message) { if (!condition) throw new Error(message); }

export function readRegular(file) {
  const resolved = path.resolve(file);
  let current = path.parse(resolved).root;
  for (const component of resolved.slice(current.length).split(path.sep)) {
    current = path.join(current, component);
    require(!fs.lstatSync(current).isSymbolicLink(), "Symlinked replay path");
  }
  const stat = fs.statSync(resolved);
  require(stat.isFile() && stat.size <= LIMIT, "Replay input must be a regular file of at most 1 MB");
  return fs.readFileSync(resolved);
}

export function load(root = ROOT) {
  const bytes = Object.fromEntries(SOURCES.map(name => [name, readRegular(path.join(root, name))]));
  const context = vm.createContext({});
  for (const name of SOURCES.filter(name => name.startsWith("public-demo/")))
    vm.runInContext(bytes[name].toString("utf8"), context, { filename: name, timeout: 5000 });
  return {
    engine: context.LavaEngine, app: context.LavaApp, fixture: context.LAVA_DEMO_DATA,
    hashes: Object.fromEntries(SOURCES.map(name => [name, digest(bytes[name])])),
  };
}

export function referenceExport(context = load()) {
  const document = context.fixture.documents[0];
  const result = context.engine.run(document, document.examples[0].query);
  return plain(context.app.buildExport(result, document, context.fixture.provenance, result.response, context.engine));
}

export function replay(payload, context = load()) {
  require(payload && payload.evidence_type === "SYNTHETIC_ONLY", "Expected a synthetic Document Desk export");
  const document = context.fixture.documents.find(item => item.id === payload.document_id);
  require(document, "Unknown authored document");
  const parameters = payload.parameters;
  require(parameters && typeof parameters === "object", "Missing retrieval settings");
  const regenerated = context.engine.run(document, payload.question, {
    topK: parameters.top_k, minimumCoverage: parameters.minimum_lexical_coverage,
  });
  // The reviewed answer is an explicit human input. All engine output and all
  // validation claims are recomputed; an edit is never relabeled as inference.
  const expected = plain(context.app.buildExport(
    regenerated, document, context.fixture.provenance, payload.reviewed_response, context.engine,
  ));
  assert.deepStrictEqual(payload, expected, "Export differs from recomputed retrieval, source text, or review validation");
  return {
    status: "verified", document_id: document.id, language: document.language,
    generated_status: regenerated.status,
    reviewed_response_modified: expected.reviewed_response_modified,
    semantic_correctness_verified: false,
    scope: "Computational replay and source-supported review; no semantic-accuracy or authenticity claim.",
  };
}

function validateOutputDirectory(output) {
  const target = path.resolve(output);
  const parent = path.dirname(target);
  require(fs.existsSync(parent) && fs.statSync(parent).isDirectory(), "Output parent must already exist");
  require(fs.realpathSync(parent) === parent, "Symlinked output parent");
  return target;
}

export function verifyBundle(output, root = ROOT) {
  const directory = validateOutputDirectory(output);
  require(fs.statSync(directory).isDirectory() && !fs.lstatSync(directory).isSymbolicLink(), "Invalid replay directory");
  assert.deepStrictEqual(fs.readdirSync(directory).sort(), ["export.json", "manifest.json", "receipt.json"]);
  const manifest = JSON.parse(readRegular(path.join(directory, "manifest.json")));
  assert.deepStrictEqual(Object.keys(manifest).sort(), ["files", "schema"]);
  require(manifest.schema === 1, "Unsupported replay manifest");
  assert.deepStrictEqual(Object.keys(manifest.files).sort(), ["export.json", "receipt.json"]);
  for (const name of ["export.json", "receipt.json"])
    require(digest(readRegular(path.join(directory, name))) === manifest.files[name], "Replay artifact hash changed: " + name);
  const context = load(root);
  const payload = JSON.parse(readRegular(path.join(directory, "export.json")));
  const receipt = JSON.parse(readRegular(path.join(directory, "receipt.json")));
  assert.deepStrictEqual(Object.keys(receipt).sort(), ["created_utc", "result", "runtime", "schema", "source_sha256"]);
  require(receipt.schema === 1 && typeof receipt.created_utc === "string" &&
    Number.isFinite(Date.parse(receipt.created_utc)) && receipt.created_utc.endsWith("Z"), "Invalid replay receipt identity");
  require(receipt.runtime === process.version, "Replay runtime changed; create a new bundle in the selected Node runtime");
  assert.deepStrictEqual(receipt.source_sha256, context.hashes, "Replay source changed");
  assert.deepStrictEqual(receipt.result, replay(payload, context), "Replay receipt claims changed");
  return receipt.result;
}

export function createBundle(output, input, root = ROOT) {
  const directory = validateOutputDirectory(output);
  if (fs.existsSync(directory)) {
    const result = verifyBundle(directory, root);
    if (input) assert.deepStrictEqual(JSON.parse(readRegular(input)), JSON.parse(readRegular(path.join(directory, "export.json"))), "Existing bundle belongs to a different input");
    return { ...result, reused: true, output: directory };
  }
  const context = load(root);
  const payload = input ? JSON.parse(readRegular(input)) : referenceExport(context);
  const result = replay(payload, context);
  const temporary = directory + ".staging-" + randomUUID();
  fs.mkdirSync(temporary);
  try {
    const exported = encode(payload);
    const receipt = encode({ schema: 1, created_utc: new Date().toISOString(), runtime: process.version, source_sha256: context.hashes, result });
    fs.writeFileSync(path.join(temporary, "export.json"), exported, { flag: "wx" });
    fs.writeFileSync(path.join(temporary, "receipt.json"), receipt, { flag: "wx" });
    fs.writeFileSync(path.join(temporary, "manifest.json"), encode({ schema: 1, files: { "export.json": digest(exported), "receipt.json": digest(receipt) } }), { flag: "wx" });
    verifyBundle(temporary, root);
    fs.renameSync(temporary, directory);
  } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
  return { ...result, reused: false, output: directory };
}

export function main(args = process.argv.slice(2)) {
  const options = {};
  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    require(["--input", "--output", "--check"].includes(key) && !Object.hasOwn(options, key), "Use --input file --output directory [--check], without duplicate flags");
    if (key === "--check") options[key] = true;
    else {
      const value = args[++i];
      require(typeof value === "string" && value && !value.startsWith("--"), "Missing argument value");
      options[key] = value;
    }
  }
  require(!options["--check"] || (options["--output"] && !options["--input"]), "--check requires --output and excludes --input");
  const output = options["--output"] ?? path.join(os.tmpdir(), "lava-replay-" + randomUUID());
  const result = options["--check"] ? verifyBundle(output) : createBundle(output, options["--input"]);
  console.log("DOCUMENT_REPLAY_VERIFIED " + JSON.stringify(result));
  return result;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); }
  catch (error) { console.error("DOCUMENT_REPLAY_FAILED: " + error.message); process.exitCode = 1; }
}
