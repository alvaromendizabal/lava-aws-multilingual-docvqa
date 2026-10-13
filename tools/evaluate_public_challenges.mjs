/** Deterministic diagnostics against the shipped demo, never a private-model benchmark. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { sources, groupKey, loadEngine, inspectCase, runCase, summarize } from "./public_challenge_adapter.mjs";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const SUITE = "examples/public_challenges.json";
export const REPORT = "reports/public_challenges.json";
export const PAGE = "reports/public_challenges.html";
export function require(condition, message) {
  if (!condition) throw new Error(message);
}
export function readSource(root, name) {
  const parts = name.split("/");
  require(!path.isAbsolute(name) && !name.includes("\\") && parts.every(p => p && p !== "." && p !== ".."), "Unsafe source path");
  require(!fs.lstatSync(root).isSymbolicLink(), "Symlinked review root");
  let current = root;
  for (const part of parts) {
    current = path.join(current, part);
    require(!fs.lstatSync(current).isSymbolicLink(), "Symlink in review source");
  }
  const stat = fs.statSync(current);
  require(stat.isFile() && stat.size <= 1_000_000, "Invalid or oversized review source");
  return fs.readFileSync(current);
}
export function validateSuite(suite, engine) {
  require(suite && Object.keys(suite).sort().join(",") === "cases,purpose,schema,scope", "Invalid suite schema");
  require(suite.schema === 1 && suite.scope === "AUTHORED_SYNTHETIC_CHALLENGES", "Invalid challenge scope");
  require(typeof suite.purpose === "string" && suite.purpose.length > 0, "Missing purpose");
  require(Array.isArray(suite.cases) && suite.cases.length > 0 && suite.cases.length <= 1000, "Invalid case count");
  const ids = new Set();
  for (const row of suite.cases) {
    require(row && typeof row === "object" && !Array.isArray(row), "Invalid challenge case");
    require(typeof row.id === "string" && /^[a-z][a-z0-9-]+$/.test(row.id) && !ids.has(row.id), "Invalid or duplicate case ID");
    ids.add(row.id);
    require(typeof row.slice === "string" && /^[a-z][a-z0-9-]+$/.test(row.slice), "Invalid slice");
    require(typeof row.query === "string" && row.query.trim().length > 0 && row.query.length <= 1200, "Invalid query");
    inspectCase(row, engine, require);
  }
  return suite;
}
export async function evaluate(root = ROOT) {
  const sourceNames = [SUITE, "tools/evaluate_public_challenges.mjs", "tools/public_challenge_adapter.mjs", ...sources].sort();
  const bytes = new Map(sourceNames.map(name => [name, readSource(root, name)]));
  const engine = await loadEngine(name => bytes.get(name).toString("utf8"));
  const suite = validateSuite(JSON.parse(bytes.get(SUITE).toString("utf8")), engine);
  const rows = suite.cases.map(row => runCase(row, engine));
  require(rows.length === suite.cases.length, "Incomplete evaluation");
  const bySlice = Object.fromEntries([...new Set(rows.map(row => row.slice))].sort().map(slice => [slice, summarize(rows.filter(row => row.slice === slice))]));
  return {
    schema: 1,
    scope: suite.scope,
    method: "Actual shipped browser engine; expected labels are used only by the evaluator.",
    selection: "Visible authored challenges, including existing UI queries and new variations. No sampling or held-out claim.",
    limitations: "These deliberately constructed cases diagnose behavior. Counts do not estimate production accuracy, neural-model quality, or statistical uncertainty.",
    sources: Object.fromEntries([...bytes].map(([name, content]) => [name, createHash("sha256").update(content).digest("hex")])),
    summary: summarize(rows),
    by_slice: bySlice,
    by_group: Object.fromEntries([...new Set(rows.map(row => row[groupKey]))].sort().map(group => [group, summarize(rows.filter(row => row[groupKey] === group))])),
    cases: rows,
  };
}
export function encode(report) { return JSON.stringify(report, null, 2) + "\n"; }
export function render(report) {
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const slices = Object.entries(report.by_slice).map(([slice, stats]) => `<tr><th scope="row">${escape(slice)}</th><td>${stats.cases}</td><td>${stats.correct}</td><td>${stats.failures}</td></tr>`).join("");
  const cases = report.cases.map(row => `<tr><th scope="row">${escape(row.id)}<small>${escape(row.slice)} · ${escape(row.policy ?? row.language)}</small></th><td>${escape(row.query)}</td><td>${escape(row.expected ?? row.expected_line ?? "abstain")}</td><td>${escape(row.decision ?? row.actual_line ?? "abstain")}</td><td><span class="${row.correct ? "match" : "mismatch"}">${row.correct ? "Match" : "Mismatch"}</span></td></tr>`).join("");
  const hashes = Object.entries(report.sources).map(([name, hash]) => `<li><span>${escape(name)}</span><code>${escape(hash)}</code></li>`).join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Public challenge evaluation · Alvaro Mendizabal</title>
<style>
:root{color-scheme:light;--ink:#152e35;--muted:#485e65;--line:#d6dfdc;--paper:#f7f7f0;--green:#195e47;--red:#943b30}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:52px 28px}a{color:var(--green);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #a85c23;outline-offset:4px}.skip{position:absolute;left:-10000px}.skip:focus{left:20px;top:10px;background:white;padding:8px}header{border-bottom:1px solid var(--line);padding-bottom:30px}.eyebrow{font-size:.8rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}h1{font:clamp(2.2rem,5vw,4rem)/1.1 Georgia,serif;letter-spacing:-.03em;margin:18px 0}h2{font:1.8rem Georgia,serif;margin:40px 0 14px}.lead{max-width:780px;color:var(--muted);font-size:1.1rem}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0}.metric{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px}.metric strong{display:block;font-size:2.3rem;line-height:1.3}.metric span{color:var(--muted)}.note{border-left:4px solid var(--green);background:#eaf1eb;padding:16px 20px}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;background:white;font-size:.9rem}th,td{text-align:left;padding:14px;vertical-align:top;border-bottom:1px solid var(--line)}thead{background:#eaf0ec}tbody tr:last-child>*{border-bottom:0}small{display:block;font-weight:400;color:var(--muted);margin-top:5px}.match,.mismatch{display:inline-block;font-weight:650;padding:3px 8px;border-radius:5px;white-space:nowrap}.match{color:var(--green);background:#e6f1e9}.mismatch{color:var(--red);background:#f8eae5}caption{text-align:left;padding:14px;font-weight:600}summary{cursor:pointer;font-weight:650}code{overflow-wrap:anywhere;font-size:.8rem}li span,li code{display:block}li{margin:12px 0}footer{margin-top:36px;color:var(--muted);font-size:.9rem}@media(max-width:650px){main{padding:32px 16px}.metrics{grid-template-columns:1fr}.metric{padding:16px}.metric strong{display:inline;margin-right:12px}.detail{min-width:850px}}@media print{body{background:white}main{padding:0;max-width:none}.scroll{overflow:visible}.detail{font-size:9pt}a{color:inherit}.metrics{grid-template-columns:repeat(3,1fr)}details{display:none}tr{break-inside:avoid}}
</style></head><body><a class="skip" href="#results">Skip to case results</a><main>
<header><p class="eyebrow">Alvaro Mendizabal / Evaluation engineering</p><h1>Test the boundary.<br>Keep the failures visible.</h1><p class="lead">A reproducible challenge review of the shipped public demo. Every decision below comes from its actual inference engine; expected outcomes enter only after inference.</p><p><a href="../README.md">Project overview</a> · <a href="../docs/PUBLIC_CHALLENGES.md">Methods &amp; interpretation</a> · <a href="public_challenges.json">Machine-readable results</a></p></header>
<div class="metrics"><div class="metric"><strong>${report.summary.cases}</strong><span>authored cases</span></div><div class="metric"><strong>${report.summary.correct}</strong><span>expected outcomes matched</span></div><div class="metric"><strong>${report.summary.failures}</strong><span>diagnostic failures retained</span></div></div>
<p class="note"><strong>Visible synthetic diagnostics.</strong> These are deliberately constructed examples, including existing demo queries. They are not a held-out benchmark, a neural-model evaluation, or an estimate of production accuracy.</p>
<h2>Performance by behavior</h2><div class="scroll"><table><caption>Counts on this authored suite</caption><thead><tr><th scope="col">Slice</th><th scope="col">Cases</th><th scope="col">Matches</th><th scope="col">Mismatches</th></tr></thead><tbody>${slices}</tbody></table></div>
<h2 id="results">Every case, including misses</h2><div class="scroll" tabindex="0" role="region" aria-label="Scrollable challenge case results"><table class="detail"><caption>Expected outcomes are authored annotations, not model inputs.</caption><thead><tr><th scope="col">Case</th><th scope="col">Query</th><th scope="col">Expected</th><th scope="col">Actual</th><th scope="col">Outcome</th></tr></thead><tbody>${cases}</tbody></table></div>
<h2>Verify these exact results</h2><p><code>node tools/evaluate_public_challenges.mjs --check</code></p><p>This recomputes the shipped engine and checks both saved files. Intentional updates require review of the cases, source fingerprints, and result differences before <code>--write</code>.</p><details><summary>Source identities (${Object.keys(report.sources).length} SHA-256 digests)</summary><ul>${hashes}</ul></details><footer>No model downloads, credentials, private documents, or external services are used. This report does not change historical research results.</footer>
</main></body></html>\n`;
}
export function verifyReport(report, expectedBytes) {
  require(encode(report) === expectedBytes.toString("utf8"), "Challenge report is stale or changed; inspect differences before --write");
}
export async function main(args = process.argv.slice(2), root = ROOT) {
  require(args.length <= 1 && (!args.length || ["--check", "--write"].includes(args[0])), "Usage: node tools/evaluate_public_challenges.mjs [--check|--write]");
  const report = await evaluate(root);
  if (args[0] === "--check") {
    verifyReport(report, readSource(root, REPORT));
    require(readSource(root, PAGE).toString("utf8") === render(report), "Challenge HTML report is stale or changed");
    console.log("CHALLENGE_REPORT_VERIFIED " + JSON.stringify(report.summary));
  } else if (args[0] === "--write") {
    for (const [name, content] of [[REPORT, encode(report)], [PAGE, render(report)]]) {
    const destination = path.join(root, name);
    // Existing reports and parent directories must pass the same symlink boundary.
    require(!fs.lstatSync(path.dirname(destination)).isSymbolicLink(), "Symlinked report directory");
    if (fs.lstatSync(destination, { throwIfNoEntry: false })) readSource(root, name);
    const temporary = `${destination}.${process.pid}.tmp`;
    try {
      fs.writeFileSync(temporary, content, { flag: "wx" });
      fs.renameSync(temporary, destination);
    } finally {
      fs.rmSync(temporary, { force: true });
    }
    }
    console.log("CHALLENGE_REPORT_WRITTEN " + JSON.stringify(report.summary));
  } else {
    process.stdout.write(encode(report));
  }
  return report;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { await main(); }
  catch (error) { console.error("CHALLENGE_EVALUATION_FAILED: " + error.message); process.exitCode = 1; }
}
