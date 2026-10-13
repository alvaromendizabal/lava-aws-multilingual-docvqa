/** Expected passages never enter retrieval or extraction; evaluation follows inference. */
import vm from "node:vm";
export const sources = ["public-demo/unicode-casefold.js", "public-demo/data.js", "public-demo/engine.js"];
export const groupKey = "language";
export async function loadEngine(read) {
  const context = vm.createContext({});
  for (const name of sources) vm.runInContext(read(name), context, { filename: name });
  return { api: context.LavaEngine, documents: context.LAVA_DEMO_DATA.documents };
}
export function inspectCase(row, engine, require) {
  const keys = Object.keys(row).filter(key => key !== "normalization").sort().join(",");
  require(keys === "document,expected_line,id,query,slice", "Unexpected challenge fields");
  require(!Object.hasOwn(row, "normalization") || row.normalization === "NFD", "Invalid normalization");
  require(row.query.length <= 500, "Oversized document query");
  const document = engine.documents.find(document => document.id === row.document);
  require(document, "Unknown document");
  engine.api.validateDocument(document);
  require(row.expected_line === null || document.pages.some(page => page.lines.some(line => line.id === row.expected_line)), "Expected passage is not in document");
}
export function runCase(row, engine) {
  const document = engine.documents.find(document => document.id === row.document);
  const query = row.normalization ? row.query.normalize(row.normalization) : row.query;
  const result = engine.api.run(document, query);
  const expectedPage = row.expected_line === null ? null : document.pages.find(page => page.lines.some(line => line.id === row.expected_line)).number;
  const rank = expectedPage === null ? null : result.retrieval.ranking.findIndex(page => page.page === expectedPage) + 1;
  const actualLine = result.evidence?.line_id ?? null;
  return {
    id: row.id, slice: row.slice, document: row.document, language: document.language, query,
    expected_line: row.expected_line, actual_line: actualLine,
    expected_page: expectedPage, actual_page: result.evidence?.page ?? null,
    status: result.status, correct: row.expected_line === actualLine,
    expected_page_rank: rank,
    schema_valid: result.validation.schema_valid,
    extractive_grounding: result.validation.extractive_grounding,
    lexical_coverage: Number(result.retrieval.lexical_coverage.toFixed(12)),
  };
}
const rate = (n, d) => d ? Number((n / d).toFixed(12)) : null;
export function summarize(rows) {
  const answerable = rows.filter(row => row.expected_line !== null);
  const unsupported = rows.filter(row => row.expected_line === null);
  const answered = rows.filter(row => row.actual_line !== null);
  return {
    cases: rows.length, correct: rows.filter(row => row.correct).length,
    failures: rows.filter(row => !row.correct).length,
    agreement: rate(rows.filter(row => row.correct).length, rows.length),
    answerable_cases: answerable.length,
    passage_matches: answerable.filter(row => row.correct).length,
    passage_agreement: rate(answerable.filter(row => row.correct).length, answerable.length),
    page_recall_at_1: rate(answerable.filter(row => row.expected_page_rank === 1).length, answerable.length),
    page_recall_at_3: rate(answerable.filter(row => row.expected_page_rank <= 3).length, answerable.length),
    page_mrr: rate(answerable.reduce((sum, row) => sum + 1 / row.expected_page_rank, 0), answerable.length),
    answered_cases: answered.length,
    answered_agreement: rate(answered.filter(row => row.correct).length, answered.length),
    unsupported_cases: unsupported.length,
    unsupported_abstentions: unsupported.filter(row => row.actual_line === null).length,
    unsupported_extractions: unsupported.filter(row => row.actual_line !== null).length,
    schema_valid_cases: rows.filter(row => row.schema_valid).length,
    grounded_cases: rows.filter(row => row.extractive_grounding).length,
  };
}
