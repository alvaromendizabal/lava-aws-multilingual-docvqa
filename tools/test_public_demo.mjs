import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const demo = path.join(root, "public-demo");
const context = vm.createContext({ console, Blob, setTimeout: (fn) => fn() });
for (const name of ["unicode-casefold.js", "data.js", "engine.js"])
  vm.runInContext(fs.readFileSync(path.join(demo, name), "utf8"), context, {
    filename: name,
  });
const engine = context.LavaEngine,
  fixture = context.LAVA_DEMO_DATA;
const plain = (value) => JSON.parse(JSON.stringify(value));
let completed = 0;
function test(name, callback) {
  callback();
  completed++;
  console.log("PASS " + name);
}

test("fixture is authored, public, multilingual and physically complete", () => {
  assert.equal(fixture.evidence_type, "SYNTHETIC_ONLY");
  assert.deepEqual(
    plain(fixture.documents.map((document) => document.language)),
    ["ja", "vi"],
  );
  for (const document of fixture.documents) {
    engine.validateDocument(document);
    assert.deepEqual(
      plain(document.pages.map((page) => page.number)),
      [1, 2, 3, 4, 5],
    );
    assert.equal(engine.pageText(document.pages[4]), "");
    assert.equal(document.pages[1].printed_label, "1");
  }
  assert.equal(fixture.provenance.model_inference, false);
  assert.equal(fixture.provenance.private_assets_used, false);
});

test("actual lexical extraction handles six independently queried facts", () => {
  const expected = [
    ["ja-2-a", "ja-3-a", "ja-4-a"],
    ["vi-2-a", "vi-3-a", "vi-4-a"],
  ];
  fixture.documents.forEach((document, docIndex) =>
    document.examples.slice(0, 3).forEach((example, queryIndex) => {
      const result = engine.run(document, example.query);
      assert.equal(result.status, "EXTRACTED");
      assert.equal(result.evidence.line_id, expected[docIndex][queryIndex]);
      assert.equal(
        result.response.answer,
        document.pages[result.evidence.page - 1].lines.find(
          (line) => line.id === result.evidence.line_id,
        ).text,
      );
      assert.equal(result.validation.schema_valid, true);
      assert.equal(result.validation.extractive_grounding, true);
      assert.equal(result.validation.semantic_correctness_verified, false);
    }),
  );
});

test("answers depend on source passages rather than supplied responses", () => {
  const document = plain(fixture.documents[0]);
  document.pages[1].lines[0].text = document.pages[1].lines[0].text.replace(
    "1,200",
    "1,347",
  );
  const result = engine.run(document, document.examples[0].query);
  assert.match(result.response.answer, /1,347/);
  assert.doesNotMatch(result.response.answer, /1,200/);
  assert.equal(result.evidence.page, 2);
});

test("empty, punctuation, OOV and low-support queries abstain without fabricated citations", () => {
  const document = fixture.documents[0];
  for (const query of ["", "?!…", "zzqxv9917", document.examples[3].query]) {
    const result = engine.run(document, query);
    assert.equal(result.status, "ABSTAINED");
    assert.equal(result.response.answer, "");
    assert.deepEqual(plain(result.response.evidence_pages), []);
  }
  const vietnamese = fixture.documents[1];
  assert.equal(
    engine.run(vietnamese, vietnamese.examples[0].query, {
      minimumCoverage: 0.7,
    }).reason,
    "INSUFFICIENT_LEXICAL_SUPPORT",
  );
  assert.equal(
    engine.run(vietnamese, vietnamese.examples[3].query).status,
    "ABSTAINED",
  );
});

test("ranking keeps textless and failed physical pages and resolves ties by page number", () => {
  const ranking = engine.retrieve(fixture.documents[0], "unfindablexyz999");
  assert.deepEqual(plain(ranking.map((row) => row.number)), [1, 2, 3, 4, 5]);
  assert.equal(ranking[4].status, "textless");
  assert.equal(ranking[4].score, 0);
  assert.throws(() => engine.run(fixture.documents[0], "x".repeat(501)), /500/);
  assert.throws(
    () => engine.run(fixture.documents[0], "東京", { topK: 0 }),
    /window/,
  );
  const altered = plain(fixture.documents[0]);
  altered.pages[4].lines.push({ id: "unverified", text: "Unverified text" });
  assert.throws(() => engine.validateDocument(altered), /unverified/);
});

const validAnswer = fixture.documents[0].pages[1].lines[0].text;
const base = {
  answer: validAnswer,
  evidence_pages: [2],
  confidence: 0.75,
  abstain: false,
};
const rawCases = [
  JSON.stringify(base),
  JSON.stringify({ ...base, evidence_pages: [2, 2] }),
  '{"answer":"","evidence_pages":[],"confidence":0,"abstain":true}',
  JSON.stringify({ ...base, confidence: true }),
  JSON.stringify({ ...base, evidence_pages: [true] }),
  JSON.stringify({ ...base, evidence_pages: ["2"] }),
  JSON.stringify({ ...base, evidence_pages: [99] }),
  JSON.stringify({ ...base, extra: 1 }),
  JSON.stringify({ ...base, abstain: true }),
  JSON.stringify({ ...base, answer: "" }),
  JSON.stringify({ ...base, confidence: "0.5" }),
  '{"answer":"text","evidence_pages":[2.0],"confidence":0.5,"abstain":false}',
  '{"answer":"text","evidence_pages":[2e0],"confidence":0.5,"abstain":false}',
  '{"answer":"text","evidence_pages":[2E+0],"confidence":1.0,"abstain":false}',
  '{"answer":"Quoted \\" is invalid JSON","evidence_pages":[2],"confidence":0.5,"abstain":false}',
  JSON.stringify({
    ...base,
    answer: 'A quoted field: "evidence_pages": [2.0] and a brace }',
    confidence: 1,
  }),
  JSON.stringify({ ...base, answer: "\u0085 trimmed \u0085" }),
  '{"answer":"text","evidence_pages":[2],"confidence":NaN,"abstain":false}',
  '{"answer":"text","evidence_pages":[2],"confidence":1e999,"abstain":false}',
  '{"answer":"text","evidence_pages":[1.0],"evidence_pages":[2],"confidence":0.5,"abstain":false}',
];
const tokens = [
  "東京拠点の出荷注文数は？",
  "Yêu cầu ưu tiên được phản hồi",
  "ＦＵＬＬ１２３ café",
  "Straße ς ﬃ",
  "Ꭰꭰ 日本語_かな 𠀀",
];
const parityInput = {
  tokens,
  raw_cases: rawCases,
  documents: fixture.documents.map((document) => ({
    pages: document.pages.map((page) => ({
      number: page.number,
      text: engine.pageText(page),
      status: page.status,
    })),
    queries: document.examples.map((example) => example.query),
  })),
};
const python = String.raw`
import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);payload=json.load(sys.stdin)
def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,root/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
lex=load('lava_demo_lexical','src/lava/retrieval/lexical.py')
parser=load('lava_demo_parser','src/lava/readers/structured_output.py')
out={'tokens':[list(lex.tokenize(text)) for text in payload['tokens']],'rankings':[],'responses':[]}
for document in payload['documents']:
    index=lex.BM25Index(tuple(lex.PageText(**page) for page in document['pages']))
    out['rankings'].append([index.rank(query) for query in document['queries']])
for raw in payload['raw_cases']:
    try:out['responses'].append({'valid':True,'value':parser.validate_structured_payload(json.loads(raw),valid_page_numbers=[1,2,3,4]).as_dict()})
    except (ValueError,TypeError,AttributeError):out['responses'].append({'valid':False})
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
`;
const compared = spawnSync(
  process.env.PYTHON || "python3",
  ["-S", "-B", "-c", python, root],
  { input: JSON.stringify(parityInput), encoding: "utf8" },
);
assert.equal(
  compared.status,
  0,
  compared.stderr || "Python public contract check failed",
);
const reference = JSON.parse(compared.stdout);

test("Unicode tokens match the actual public Python tokenizer", () => {
  tokens.forEach((text, index) =>
    assert.deepEqual(plain(engine.tokenize(text)), reference.tokens[index]),
  );
});
test("all eight page rankings numerically match the actual Python BM25", () => {
  fixture.documents.forEach((document, docIndex) =>
    document.examples.forEach((example, queryIndex) => {
      const result = engine.retrieve(document, example.query);
      result.forEach((row, index) => {
        assert.equal(
          row.number,
          reference.rankings[docIndex][queryIndex][index][0],
        );
        assert.ok(
          Math.abs(
            row.score - reference.rankings[docIndex][queryIndex][index][1],
          ) < 1e-10,
        );
      });
    }),
  );
});
test("raw response validation matches Python, including decimal and exponent citation rejection", () => {
  rawCases.forEach((raw, index) => {
    let response;
    try {
      response = {
        valid: true,
        value: plain(engine.parseResponse(raw, [1, 2, 3, 4])),
      };
    } catch (_) {
      response = { valid: false };
    }
    assert.deepEqual(
      response,
      reference.responses[index],
      `Raw contract case ${index}: ${raw}`,
    );
  });
  assert.throws(
    () => engine.parseResponse("[".repeat(18) + "0" + "]".repeat(18), [2]),
    /deep/,
  );
  assert.throws(() => engine.parseResponse(" ".repeat(65537), [2]), /64 KiB/);
});
test("schema acceptance and exact-source support remain distinct", () => {
  const validation = engine.validateEvidence(
    { ...base, answer: "An invented plausible answer." },
    fixture.documents[0],
    [2],
  );
  assert.equal(validation.schema_valid, true);
  assert.equal(validation.extractive_grounding, false);
  assert.equal(validation.semantic_correctness_verified, false);
  assert.equal(
    engine.validateEvidence(base, fixture.documents[0], [2])
      .extractive_grounding,
    true,
  );
});

class Element {
  constructor(id) {
    this.id = id;
    this.value = "";
    this.textContent = "";
    this.innerHTML = "";
    this.className = "";
    this.dataset = {};
    this.listeners = {};
    this.attributes = {};
    this.disabled = false;
    this.children = [];
  }
  addEventListener(type, callback) {
    (this.listeners[type] ||= []).push(callback);
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  getAttribute(name) {
    return this.attributes[name];
  }
  appendChild(child) {
    this.children.push(child);
    child.parent = this;
    return child;
  }
  remove() {
    if (this.parent)
      this.parent.children = this.parent.children.filter(
        (item) => item !== this,
      );
  }
  focus() {
    this.focused = true;
  }
  click() {
    this.clicked = true;
    this.fire("click");
  }
  closest(selector) {
    const match = selector.match(/^\[data-(.+)\]$/);
    return match && Object.hasOwn(this.dataset, match[1]) ? this : null;
  }
  fire(type, extra = {}) {
    if (type === "click" && this.disabled) return;
    const event = {
      target: this,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...extra,
    };
    for (const listener of this.listeners[type] || []) listener(event);
    return event;
  }
}
const html = fs.readFileSync(path.join(demo, "index.html"), "utf8");
const elements = new Map(
  [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [
    match[1],
    new Element(match[1]),
  ]),
);
const document = {
  getElementById: (id) => elements.get(id),
  body: new Element("body"),
  createElement: (tag) => new Element(tag),
};
const el = (id) => {
  assert.ok(elements.has(id), "Actual HTML must contain " + id);
  return elements.get(id);
};
el("top-k").value = "2";
el("support-threshold").value = "0.20";
const downloads = [],
  revoked = [];
context.document = document;
context.URL = {
  createObjectURL(blob) {
    downloads.push(blob);
    return "blob:fixture-" + downloads.length;
  },
  revokeObjectURL(url) {
    revoked.push(url);
  },
};
vm.runInContext(fs.readFileSync(path.join(demo, "app.js"), "utf8"), context, {
  filename: "app.js",
});
const app = context.lavaDemo;
assert.ok(
  app,
  el("run-status").textContent || "Actual application did not mount",
);
function clickData(container, key, value) {
  const target = new Element("button");
  target.dataset[key] = String(value);
  el(container).fire("click", { target });
}

test("actual bootstrap retrieves Japanese evidence and renders its physical citation", () => {
  assert.equal(app.state.result.evidence.line_id, "ja-2-a");
  assert.equal(app.state.page, 2);
  assert.match(el("paper").innerHTML, /SELECTED EVIDENCE/);
  assert.match(el("citation-list").innerHTML, /Physical p\. 2 · Printed 1/);
  assert.equal(el("paper").attributes.lang, "ja");
  assert.match(el("rankings").innerHTML, /No native text/);
});
test("actual document, example, page-navigation and unavailable-page handlers work", () => {
  clickData("document-list", "document", "an-phuc-support");
  assert.equal(app.state.result.evidence.line_id, "vi-2-a");
  el("previous-page").fire("click");
  assert.equal(app.state.page, 1);
  el("previous-page").fire("click");
  assert.equal(app.state.page, 1);
  el("next-page").fire("click");
  assert.equal(app.state.page, 2);
  clickData("page-tabs", "page", 4);
  assert.equal(app.state.page, 4);
  el("unavailable-button").fire("click");
  assert.equal(app.state.page, 5);
  assert.match(el("paper").innerHTML, /Simulated extraction failure/);
  assert.equal(el("next-page").disabled, true);
  clickData("citation-list", "page", 2);
  assert.equal(app.state.page, 2);
  assert.equal(el("paper").focused, true);
  clickData("examples", "query", 2);
  assert.equal(app.state.result.evidence.line_id, "vi-4-a");
});
test("edited queries invalidate export until actual form or keyboard rerun", () => {
  el("question").value = fixture.documents[1].examples[0].query;
  el("question").fire("input");
  assert.equal(el("answer-state").textContent, "Stale result");
  assert.equal(el("export-result").disabled, true);
  el("validate-response").fire("click");
  assert.match(el("validation-result").innerHTML, /Run retrieval/);
  el("export-result").fire("click");
  assert.equal(downloads.length, 0);
  const event = el("question").fire("keydown", { key: "Enter", ctrlKey: true });
  assert.equal(event.defaultPrevented, true);
  assert.equal(app.state.result.evidence.line_id, "vi-2-a");
  assert.equal(el("export-result").disabled, false);
  el("question").value = "";
  el("query-form").fire("submit");
  assert.equal(app.state.result.reason, "EMPTY_QUERY");
  assert.equal(el("citation-list").innerHTML, "");
  clickData("examples", "query", 0);
});
test("actual retrieval controls recompute support and can cause abstention", () => {
  el("support-threshold").value = ".7";
  el("support-threshold").fire("input");
  assert.equal(el("threshold-value").textContent, "70%");
  assert.equal(el("export-result").disabled, true);
  el("support-threshold").fire("change");
  assert.equal(app.state.result.status, "ABSTAINED");
  el("support-threshold").value = ".2";
  el("support-threshold").fire("input");
  el("support-threshold").fire("change");
  el("top-k").value = "1";
  el("top-k").fire("change");
  assert.equal(app.state.result.retrieval.supplied_pages.length, 1);
  clickData("examples", "query", 3);
  assert.equal(app.state.result.status, "ABSTAINED");
  clickData("examples", "query", 0);
});
test("actual response editor rejects invalid citations and unsupported text", () => {
  el("bad-citation").fire("click");
  assert.match(el("validation-result").innerHTML, /page 99/);
  el("export-result").fire("click");
  assert.equal(downloads.length, 0);
  el("reset-response").fire("click");
  assert.match(el("validation-result").innerHTML, /exact passage/);
  el("unsupported-answer").fire("click");
  assert.match(el("validation-result").innerHTML, /Schema accepted/);
  el("export-result").fire("click");
  assert.equal(downloads.length, 0);
  el("response-json").value =
    '{"answer":"x","evidence_pages":[2.0],"confidence":0.5,"abstain":false}';
  el("validate-response").fire("click");
  assert.match(el("validation-result").innerHTML, /JSON integers/);
  el("response-json").value = "{broken";
  el("validate-response").fire("click");
  assert.match(el("validation-result").innerHTML, /invalid/i);
  el("reset-response").fire("click");
});
test("actual export contains original and reviewed evidence with validation state", () => {
  const edited = plain(app.state.result.response);
  edited.confidence = 0.123;
  el("response-json").value = JSON.stringify(edited);
  el("export-result").fire("click");
  assert.equal(downloads.length, 1);
  assert.equal(revoked.length, 1);
  assert.equal(app.state.lastExport.reviewed_response.confidence, 0.123);
  assert.notEqual(app.state.lastExport.response.confidence, 0.123);
  assert.equal(app.state.lastExport.reviewed_response_modified, true);
  assert.equal(
    app.state.lastExport.reviewed_validation.extractive_grounding,
    true,
  );
  assert.equal(app.state.lastExport.evidence_type, "SYNTHETIC_ONLY");
});
const exported = JSON.parse(await downloads[0].text());
assert.deepEqual(exported, plain(app.state.lastExport));

test("runtime errors clear stale content and disable export, then recover", () => {
  el("top-k").value = "0";
  el("top-k").fire("change");
  assert.equal(app.state.result, null);
  assert.equal(el("answer-state").textContent, "Error");
  assert.equal(el("answer-text").textContent, "");
  assert.equal(el("export-result").disabled, true);
  el("top-k").value = "2";
  el("top-k").fire("change");
  assert.equal(app.state.result.status, "EXTRACTED");
  assert.throws(
    () =>
      context.LavaApp.mount(
        document,
        { evidence_type: "SYNTHETIC_ONLY", documents: [] },
        engine,
      ),
    /missing/,
  );
});
test("source rendering escapes authored or edited text instead of executing HTML", () => {
  const altered = plain(fixture);
  altered.documents[0].pages[1].lines[0].text =
    "<img src=x onerror=alert(1)> 東京拠点の出荷注文数は1,200件です。";
  const another = context.LavaApp.mount(document, altered, engine);
  assert.equal(another.state.result.status, "EXTRACTED");
  assert.doesNotMatch(el("paper").innerHTML, /<img/);
  assert.match(el("paper").innerHTML, /&lt;img/);
});

function contrast(a, b) {
  const lum = (hex) => {
    const rgb = hex
      .match(/[a-f\d]{2}/gi)
      .map((value) => parseInt(value, 16) / 255)
      .map((value) =>
        value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
      );
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  };
  const x = lum(a),
    y = lum(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
test("buildless assets, semantic controls and accessible baseline styling are present", () => {
  assert.match(html, /<html lang="en">/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /<label\s+for="question"\s*>/);
  assert.match(html, /<label\s+for="response-json"\s*>/);
  assert.match(html, /<noscript\s*>/);
  const scripts = [...html.matchAll(/<script[^>]*src="([^"]+)"/g)].map(
    (match) => match[1],
  );
  for (const source of scripts) {
    assert.ok(!source.includes("://"));
    assert.ok(fs.existsSync(path.join(demo, source)));
  }
  const css = fs.readFileSync(path.join(demo, "styles.css"), "utf8");
  assert.match(css, /focus-visible/);
  assert.match(css, /prefers-reduced-motion/);
  assert.match(css, /max-width:\s*440px/);
  assert.doesNotMatch(css, /font-size:\s*(?:[0-9]|1[01])px\b/);
  for (const [foreground, background] of [
    ["243643", "fffefb"],
    ["58656a", "fffefb"],
    ["9d422b", "f5f1e9"],
    ["1e644e", "e9f3ed"],
    ["755023", "f7ecd2"],
  ])
    assert.ok(contrast(foreground, background) >= 4.5);
  for (const source of ["engine.js", "app.js"])
    assert.doesNotMatch(
      fs.readFileSync(path.join(demo, source), "utf8"),
      /\bfetch\s*\(|XMLHttpRequest|WebSocket|eval\s*\(/,
    );
});

console.log(
  JSON.stringify(
    {
      status: "PASS",
      tests: completed,
      scope:
        "Actual engine and UI event handlers with fake DOM; Python tokenizer/BM25/raw-validator parity. No browser screenshot or layout execution claimed.",
    },
    null,
    2,
  ),
);
