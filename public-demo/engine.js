/* Browser-local BM25 and extractive evidence. No model calls or supplied answers. */
(function (root) {
  "use strict";
  const schema = "lava-public-evidence-run-v1";
  function need(condition, message) {
    if (!condition) throw new Error(message);
  }
  function foldCase(text) {
    const overrides = root.LAVA_CASEFOLD || {};
    return Array.from(
      String(text).normalize("NFKC"),
      (char) => overrides[char] ?? char.toLowerCase(),
    ).join("");
  }
  function tokenize(text) {
    const segments = foldCase(text).match(/[\p{L}\p{N}]+/gu) || [];
    const tokens = [];
    for (const segment of segments) {
      tokens.push("w:" + segment);
      const chars = Array.from(segment);
      for (const size of [2, 3])
        for (let start = 0; start <= chars.length - size; start++)
          tokens.push(`c${size}:` + chars.slice(start, start + size).join(""));
    }
    return tokens;
  }
  function codePointOrder(a, b) {
    const left = Array.from(a, (c) => c.codePointAt(0)),
      right = Array.from(b, (c) => c.codePointAt(0));
    for (let i = 0; i < Math.min(left.length, right.length); i++)
      if (left[i] !== right[i]) return left[i] - right[i];
    return left.length - right.length;
  }
  function counts(tokens) {
    const out = new Map();
    for (const token of tokens) out.set(token, (out.get(token) || 0) + 1);
    return out;
  }
  function bm25(rows, question, { k1 = 1.2, b = 0.75 } = {}) {
    need(rows.length > 0, "At least one indexed position is required.");
    need(
      Number.isFinite(k1) && k1 > 0 && Number.isFinite(b) && b >= 0 && b <= 1,
      "Invalid BM25 parameters.",
    );
    const termCounts = rows.map((row) => counts(tokenize(row.text))),
      lengths = termCounts.map((count) =>
        [...count.values()].reduce((a, n) => a + n, 0),
      );
    const average = lengths.reduce((a, n) => a + n, 0) / rows.length,
      frequencies = new Map();
    for (const count of termCounts)
      for (const token of count.keys())
        frequencies.set(token, (frequencies.get(token) || 0) + 1);
    const terms = [...new Set(tokenize(question))].sort(codePointOrder);
    return rows
      .map((row, index) => {
        let score = 0;
        if (average) {
          const normalization = k1 * (1 - b + (b * lengths[index]) / average);
          for (const token of terms) {
            const frequency = termCounts[index].get(token) || 0;
            if (frequency) {
              const df = frequencies.get(token);
              score +=
                (Math.log1p((rows.length - df + 0.5) / (df + 0.5)) *
                  frequency *
                  (k1 + 1)) /
                (frequency + normalization);
            }
          }
        }
        return { ...row, score };
      })
      .sort((a, b) => b.score - a.score || a.number - b.number);
  }
  function pageText(page) {
    return page.status === "ok"
      ? [page.title, ...page.lines.map((line) => line.text)].join("\n")
      : "";
  }
  function validateDocument(document) {
    need(
      document &&
        typeof document.id === "string" &&
        Array.isArray(document.pages) &&
        document.pages.length > 0,
      "Document pages are missing.",
    );
    const lineIds = new Set();
    for (const [index, page] of document.pages.entries()) {
      need(
        page.number === index + 1,
        "Physical pages must remain consecutive and one-indexed.",
      );
      need(
        ["ok", "textless", "extraction_error"].includes(page.status),
        "Unknown extraction status.",
      );
      need(
        Array.isArray(page.lines) &&
          (page.status === "ok" || page.lines.length === 0),
        "Unavailable pages cannot carry unverified text.",
      );
      for (const line of page.lines) {
        need(
          typeof line.id === "string" &&
            !lineIds.has(line.id) &&
            typeof line.text === "string" &&
            line.text.trim(),
          "Invalid or duplicate passage.",
        );
        lineIds.add(line.id);
      }
    }
    return document;
  }
  const pyTrim = (text) =>
    text.replace(
      /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/gu,
      "",
    );
  function validatePayload(payload, validPages) {
    need(
      payload && typeof payload === "object" && !Array.isArray(payload),
      "Response must be a JSON object.",
    );
    need(
      Object.keys(payload).sort().join(",") ===
        "abstain,answer,confidence,evidence_pages",
      "Response must contain exactly answer, evidence_pages, confidence and abstain.",
    );
    need(typeof payload.answer === "string", "answer must be a string.");
    need(typeof payload.abstain === "boolean", "abstain must be a boolean.");
    need(
      typeof payload.confidence === "number" &&
        Number.isFinite(payload.confidence) &&
        payload.confidence >= 0 &&
        payload.confidence <= 1,
      "confidence must be a finite number in [0, 1].",
    );
    need(
      Array.isArray(payload.evidence_pages),
      "evidence_pages must be an array.",
    );
    const allowed = validPages === undefined ? null : new Set(validPages),
      pages = [];
    for (const page of payload.evidence_pages) {
      need(
        Number.isInteger(page) && page > 0,
        "Evidence pages must be positive integers.",
      );
      need(
        allowed === null || allowed.has(page),
        `Physical page ${page} is not in the supplied evidence.`,
      );
      if (!pages.includes(page)) pages.push(page);
    }
    const answer = pyTrim(payload.answer);
    need(
      !payload.abstain || (!answer && pages.length === 0),
      "Abstention requires an empty answer and no citations.",
    );
    need(
      payload.abstain || answer.length > 0,
      "A non-abstaining response requires an answer.",
    );
    return {
      answer,
      evidence_pages: pages,
      confidence: payload.confidence,
      abstain: payload.abstain,
    };
  }
  function parseResponse(raw, validPages) {
    // Preserve JSON integer syntax before JavaScript erases the distinction
    // between 1, 1.0 and 1e0. Only the first is an integer in Python json.loads.
    need(
      typeof raw === "string" && raw.length <= 65536,
      "Response JSON must be text of at most 64 KiB.",
    );
    let position = 0;
    function whitespace() {
      while (position < raw.length && " \n\r\t".includes(raw[position]))
        position++;
    }
    function string() {
      const start = position++;
      let escaped = false;
      while (position < raw.length) {
        const char = raw[position++];
        if (char === '"' && !escaped)
          return JSON.parse(raw.slice(start, position));
        if (char === "\\" && !escaped) escaped = true;
        else escaped = false;
      }
      throw new Error("Unterminated JSON string.");
    }
    function value(depth = 0) {
      need(depth <= 16, "Response JSON is nested too deeply.");
      whitespace();
      const char = raw[position];
      if (char === '"') return { value: string(), type: "string" };
      if (char === "{" || char === "[") {
        const array = char === "[",
          end = array ? "]" : "}";
        const values = array ? [] : Object.create(null),
          children = array ? [] : Object.create(null);
        position++;
        whitespace();
        if (raw[position] === end) {
          position++;
          return { value: values, children, type: array ? "array" : "object" };
        }
        while (position < raw.length) {
          let key = values.length;
          if (!array) {
            need(raw[position] === '"', "JSON object keys must be quoted.");
            key = string();
            whitespace();
            need(raw[position++] === ":", "Missing JSON colon.");
          }
          const child = value(depth + 1);
          values[key] = child.value;
          children[key] = child;
          whitespace();
          const separator = raw[position++];
          if (separator === end)
            return {
              value: values,
              children,
              type: array ? "array" : "object",
            };
          need(separator === ",", "Invalid JSON separator.");
          whitespace();
        }
        throw new Error("Unterminated JSON collection.");
      }
      for (const [literal, result] of [
        ["true", true],
        ["false", false],
        ["null", null],
      ]) {
        if (raw.startsWith(literal, position)) {
          position += literal.length;
          return { value: result, type: typeof result };
        }
      }
      const number = raw
        .slice(position)
        .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      need(Boolean(number), "Invalid JSON value.");
      position += number[0].length;
      return {
        value: Number(number[0]),
        type: "number",
        integerSyntax: !/[.eE]/.test(number[0]),
      };
    }
    const parsed = value();
    whitespace();
    need(position === raw.length, "Unexpected trailing JSON content.");
    const pageNodes = parsed.children?.evidence_pages;
    if (pageNodes?.type === "array")
      for (const page of pageNodes.children) {
        if (page.type === "number")
          need(
            page.integerSyntax,
            "Evidence page tokens must be JSON integers, not decimals or exponents.",
          );
      }
    return validatePayload(parsed.value, validPages);
  }
  function validateEvidence(payload, document, suppliedPages) {
    try {
      const parsed = validatePayload(payload, suppliedPages);
      const grounded =
        parsed.abstain ||
        parsed.evidence_pages.some((number) =>
          document.pages[number - 1].lines.some(
            (line) => line.text === parsed.answer,
          ),
        );
      return {
        schema_valid: true,
        extractive_grounding: grounded,
        abstention_consistent: parsed.abstain,
        semantic_correctness_verified: false,
        parsed,
        message: parsed.abstain
          ? "Valid abstention: no answer or citation asserted."
          : grounded
            ? "Valid schema. The answer is an exact passage from a supplied physical page."
            : "Schema accepted, but the answer is not an exact passage on a cited page.",
      };
    } catch (error) {
      return {
        schema_valid: false,
        extractive_grounding: false,
        semantic_correctness_verified: false,
        message: error.message,
      };
    }
  }
  function validateResponseText(raw, document, suppliedPages) {
    try {
      return validateEvidence(
        parseResponse(raw, suppliedPages),
        document,
        suppliedPages,
      );
    } catch (error) {
      return {
        schema_valid: false,
        extractive_grounding: false,
        semantic_correctness_verified: false,
        message: error.message,
      };
    }
  }
  function retrieve(document, question) {
    validateDocument(document);
    return bm25(
      document.pages.map((page) => ({
        number: page.number,
        text: pageText(page),
        status: page.status,
      })),
      question,
    );
  }
  function run(document, question, { topK = 2, minimumCoverage = 0.2 } = {}) {
    validateDocument(document);
    need(
      typeof question === "string" && question.length <= 500,
      "Use a question of at most 500 characters.",
    );
    need(
      Number.isInteger(topK) && topK >= 1 && topK <= document.pages.length,
      "Invalid evidence window.",
    );
    need(
      Number.isFinite(minimumCoverage) &&
        minimumCoverage > 0 &&
        minimumCoverage <= 1,
      "Invalid lexical support threshold.",
    );
    const queryTokens = [...new Set(tokenize(question))],
      ranking = retrieve(document, question);
    const supplied = ranking
      .filter((row) => row.score > 0 && row.status === "ok")
      .slice(0, topK)
      .map((row) => row.number);
    const passages = supplied.flatMap((number) =>
      document.pages[number - 1].lines.map((line, index) => ({
        ...line,
        number: number * 1000 + index,
        page: number,
      })),
    );
    const rankedPassages = passages.length ? bm25(passages, question) : [];
    const best = rankedPassages[0] || null,
      bestTerms = new Set(best ? tokenize(best.text) : []);
    const matched = queryTokens.filter((token) => bestTerms.has(token));
    const coverage = queryTokens.length
      ? matched.length / queryTokens.length
      : 0;
    let reason = null;
    if (!question.trim()) reason = "EMPTY_QUERY";
    else if (!queryTokens.length) reason = "NO_SEARCHABLE_TERMS";
    else if (!best || best.score <= 0) reason = "NO_MATCHING_EVIDENCE";
    else if (coverage < minimumCoverage)
      reason = "INSUFFICIENT_LEXICAL_SUPPORT";
    const response = {
      answer: reason ? "" : best.text,
      evidence_pages: reason ? [] : [best.page],
      confidence: reason ? 0 : coverage,
      abstain: Boolean(reason),
    };
    const validation = validateEvidence(response, document, supplied);
    need(
      validation.schema_valid && validation.extractive_grounding,
      "Generated evidence failed validation.",
    );
    return {
      schema,
      evidence_type: "SYNTHETIC_ONLY",
      document_id: document.id,
      question,
      parameters: {
        top_k: topK,
        minimum_lexical_coverage: minimumCoverage,
        bm25_k1: 1.2,
        bm25_b: 0.75,
      },
      status: reason ? "ABSTAINED" : "EXTRACTED",
      reason,
      response,
      validation,
      evidence: reason
        ? null
        : {
            page: best.page,
            line_id: best.id,
            text: best.text,
            passage_score: best.score,
          },
      retrieval: {
        ranking: ranking.map(({ number, score, status }) => ({
          page: number,
          score,
          status,
        })),
        supplied_pages: supplied,
        query_terms: queryTokens.length,
        matched_terms: matched.length,
        lexical_coverage: coverage,
      },
      limitations: [
        "Synthetic source documents, not a benchmark.",
        "Lexical matching and verbatim extraction; no language model, OCR, translation or multi-page reasoning.",
        "Confidence is uncalibrated lexical coverage, not answer probability.",
        "Valid citations and exact copying do not establish that a passage answers the question.",
      ],
    };
  }
  const api = {
    schema,
    tokenize,
    bm25,
    pageText,
    validateDocument,
    validatePayload,
    parseResponse,
    validateEvidence,
    validateResponseText,
    retrieve,
    run,
  };
  root.LavaEngine = Object.freeze(api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
