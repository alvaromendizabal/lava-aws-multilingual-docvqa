(function (root) {
  "use strict";
  const escape = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (char) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[char],
    );
  const reasonText = {
    EMPTY_QUERY: "Enter a question to search this document.",
    NO_SEARCHABLE_TERMS: "The query contains no searchable letters or numbers.",
    NO_MATCHING_EVIDENCE:
      "No indexed passage matches this question. No answer or citation is asserted.",
    INSUFFICIENT_LEXICAL_SUPPORT:
      "The best passage does not meet the selected lexical coverage threshold. The demo abstains.",
  };
  function mount(
    document,
    fixture = root.LAVA_DEMO_DATA,
    engine = root.LavaEngine,
  ) {
    if (
      !fixture ||
      fixture.evidence_type !== "SYNTHETIC_ONLY" ||
      !fixture.documents?.length ||
      !engine
    )
      throw new Error("Synthetic fixture or retrieval engine is missing.");
    fixture.documents.forEach(engine.validateDocument);
    const $ = (id) => {
      const element = document.getElementById(id);
      if (!element) throw new Error("Missing interface element: " + id);
      return element;
    };
    const state = {
      documentId: fixture.documents[0].id,
      page: 1,
      result: null,
      lastExport: null,
    };
    const currentDocument = () =>
      fixture.documents.find((item) => item.id === state.documentId);
    function isStale() {
      return Boolean(
        state.result &&
          (state.result.question !== $("question").value ||
            state.result.parameters.top_k !== Number($("top-k").value) ||
            state.result.parameters.minimum_lexical_coverage !==
              Number($("support-threshold").value)),
      );
    }
    function markStale() {
      if (!isStale()) return;
      $("answer-state").textContent = "Stale result";
      $("answer-state").className = "state-tag abstained";
      $("run-status").textContent =
        "Question or controls changed. Run retrieval to refresh this result before validating or exporting.";
      $("export-result").disabled = true;
    }
    function renderLibrary() {
      $("document-list").innerHTML = fixture.documents
        .map(
          (item) =>
            `<button type="button" class="document-card" data-document="${escape(item.id)}" aria-pressed="${item.id === state.documentId}"><span class="file-icon" aria-hidden="true"></span><span class="document-language">${escape(item.language_label)} · ${item.pages.length} pages</span><strong lang="${escape(item.language)}">${escape(item.title)}</strong><span class="document-description">${escape(item.subtitle)}</span></button>`,
        )
        .join("");
      $("examples").innerHTML = currentDocument()
        .examples.map(
          (item, index) =>
            `<button type="button" class="example-button" data-query="${index}">${escape(item.label)}</button>`,
        )
        .join("");
      $("question").setAttribute("lang", currentDocument().language);
      $("viewer-language").textContent =
        currentDocument().language_label + " / Synthetic document";
      $("viewer-title").textContent = currentDocument().title;
    }
    function renderPage() {
      const selected = currentDocument(),
        page = selected.pages[state.page - 1],
        evidence = state.result?.evidence;
      const lines =
        page.status === "ok"
          ? page.lines
              .map((line) => {
                const isEvidence =
                  evidence?.page === page.number &&
                  evidence.line_id === line.id;
                return `<p class="source-line${line.label ? " has-label" : ""}${isEvidence ? " selected" : ""}" id="passage-${escape(line.id)}">${isEvidence ? '<span class="evidence-pin" lang="en">SELECTED EVIDENCE</span>' : ""}${line.label ? `<span class="row-label">${escape(line.label)}</span>` : ""}${escape(line.text)}</p>`;
              })
              .join("")
          : `<div class="unavailable-box" lang="en"><div class="warehouse" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span></div><strong>${page.status === "textless" ? "No indexed native text" : "Simulated extraction failure"}</strong><p>${escape(page.display_note)}</p></div>`;
      $("paper").setAttribute("lang", selected.language);
      $("paper").innerHTML =
        `<div class="paper-topline"><span>${escape(selected.code)}</span><span lang="en">AUTHORED SYNTHETIC</span></div><p class="paper-section">${escape(page.section)}</p><h3>${escape(page.title)}</h3><div class="paper-lines">${lines}</div><div class="paper-stamp" lang="en">DEMONSTRATION DOCUMENT</div><div class="paper-bottom" lang="en"><span>Physical page ${page.number} of ${selected.pages.length}</span><span>Printed: ${escape(page.printed_label)}</span></div>`;
      $("page-position").textContent =
        `${state.page} / ${selected.pages.length}`;
      $("previous-page").disabled = state.page === 1;
      $("next-page").disabled = state.page === selected.pages.length;
      $("page-status").textContent = {
        ok: "Native text indexed",
        textless: "Textless · not indexed",
        extraction_error: "Extraction error · not indexed",
      }[page.status];
      $("page-tabs").innerHTML = selected.pages
        .map(
          (item) =>
            `<button type="button" class="page-tab${item.status !== "ok" ? " unavailable" : ""}" data-page="${item.number}" aria-label="Physical page ${item.number}${item.status !== "ok" ? ", unavailable text" : ""}"${item.number === state.page ? ' aria-current="page"' : ""}>${item.number}</button>`,
        )
        .join("");
    }
    function showValidation(validation) {
      const type = !validation.schema_valid
        ? "invalid"
        : !validation.extractive_grounding
          ? "warning"
          : "valid";
      $("validation-result").innerHTML =
        `<div class="validation-message ${type}">${escape(validation.message)}</div>`;
    }
    function validateEditor() {
      if (!state.result) return null;
      const validation = isStale()
        ? {
            schema_valid: false,
            extractive_grounding: false,
            message:
              "Run retrieval for the current question and controls before validating or exporting.",
          }
        : engine.validateResponseText(
            $("response-json").value,
            currentDocument(),
            state.result.retrieval.supplied_pages,
          );
      showValidation(validation);
      return validation;
    }
    function restoreEditor() {
      if (!state.result) return;
      $("response-json").value = JSON.stringify(state.result.response, null, 2);
      validateEditor();
    }
    function renderResult() {
      const result = state.result,
        selected = currentDocument();
      $("answer-state").textContent =
        result.status === "EXTRACTED" ? "Extracted" : "Abstained";
      $("answer-state").className =
        "state-tag" + (result.status === "ABSTAINED" ? " abstained" : "");
      $("run-status").textContent = result.reason
        ? reasonText[result.reason]
        : `Copied one passage from ${result.retrieval.supplied_pages.length} retrieved physical page${result.retrieval.supplied_pages.length === 1 ? "" : "s"}.`;
      $("answer-text").textContent = result.response.answer;
      $("answer-text").setAttribute("lang", selected.language);
      $("citation-list").innerHTML = result.response.evidence_pages
        .map(
          (number) =>
            `<button type="button" class="citation-button" data-page="${number}">Physical p. ${number} · Printed ${escape(selected.pages[number - 1].printed_label)} <span aria-hidden="true">↗</span></button>`,
        )
        .join("");
      $("support-readout").textContent =
        `${Math.round(result.retrieval.lexical_coverage * 100)}% lexical coverage · ${result.retrieval.matched_terms} / ${result.retrieval.query_terms} unique query features`;
      const maximum = result.retrieval.ranking[0]?.score || 1;
      $("rankings").innerHTML = result.retrieval.ranking
        .map((row) => {
          const page = selected.pages[row.page - 1],
            supplied = result.retrieval.supplied_pages.includes(row.page);
          return `<button type="button" class="rank-row${supplied ? " in-window" : ""}" data-page="${row.page}" aria-label="View physical page ${row.page}, BM25 score ${row.score.toFixed(2)}${supplied ? ", supplied evidence" : ""}"><span class="rank-page">${row.page}</span><span><span class="rank-name" lang="${escape(selected.language)}">${escape(page.title)}</span>${row.status === "ok" ? `<span class="rank-bar" aria-hidden="true"><span style="width:${((row.score / maximum) * 100).toFixed(2)}%"></span></span>` : `<span class="rank-unavailable">${row.status === "textless" ? "No native text" : "Extraction error"}</span>`}</span><span class="rank-score">${row.score.toFixed(2)}</span></button>`;
        })
        .join("");
      $("export-result").disabled = false;
      restoreEditor();
    }
    function execute() {
      $("query-count").textContent = `${$("question").value.length} / 500`;
      try {
        state.result = engine.run(currentDocument(), $("question").value, {
          topK: Number($("top-k").value),
          minimumCoverage: Number($("support-threshold").value),
        });
        if (state.result.evidence) state.page = state.result.evidence.page;
        renderResult();
        renderPage();
      } catch (error) {
        state.result = null;
        $("answer-state").textContent = "Error";
        $("answer-state").className = "state-tag error";
        $("run-status").textContent = error.message;
        $("answer-text").textContent = "";
        $("citation-list").innerHTML = "";
        $("support-readout").textContent = "";
        $("rankings").innerHTML = "";
        $("response-json").value = "";
        $("export-result").disabled = true;
        showValidation({
          schema_valid: false,
          extractive_grounding: false,
          message: "No result is available to validate or export.",
        });
        renderPage();
      }
      return state.result;
    }
    function goToPage(number, focus = false) {
      if (
        !Number.isInteger(number) ||
        number < 1 ||
        number > currentDocument().pages.length
      )
        return;
      state.page = number;
      renderPage();
      if (focus) $("paper").focus({ preventScroll: true });
    }
    function selectDocument(id) {
      if (!fixture.documents.some((item) => item.id === id)) return;
      state.documentId = id;
      state.page = 1;
      state.result = null;
      $("question").value = currentDocument().examples[0].query;
      renderLibrary();
      execute();
    }
    function exportResult() {
      const validation = validateEditor();
      if (!validation?.schema_valid || !validation.extractive_grounding) {
        showValidation({
          schema_valid: false,
          message:
            "Export blocked: first provide a structurally valid, source-supported extract or consistent abstention.",
        });
        return null;
      }
      const payload = {
        ...state.result,
        reviewed_response: validation.parsed,
        reviewed_validation: validation,
        reviewed_response_modified:
          JSON.stringify(validation.parsed) !==
          JSON.stringify(state.result.response),
        export_scope:
          "Original retrieval result plus a separate reviewed response. Edited responses are not relabeled as generated extracts.",
        provenance: fixture.provenance,
        source_document: {
          id: currentDocument().id,
          title: currentDocument().title,
          physical_page_count: currentDocument().pages.length,
          evidence_pages: state.result.retrieval.supplied_pages.map(
            (number) => ({
              number,
              text: engine.pageText(currentDocument().pages[number - 1]),
            }),
          ),
        },
      };
      const raw = JSON.stringify(payload, null, 2);
      if (!root.Blob || !root.URL?.createObjectURL) {
        showValidation({
          schema_valid: false,
          message:
            "This browser cannot create a local download. The response remains visible for copying.",
        });
        return null;
      }
      const blob = new root.Blob([raw], {
          type: "application/json;charset=utf-8",
        }),
        url = root.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `lava-evidence-${state.documentId}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      root.setTimeout(() => root.URL.revokeObjectURL(url), 0);
      state.lastExport = payload;
      return payload;
    }
    $("document-list").addEventListener("click", (event) => {
      const item = event.target.closest("[data-document]");
      if (item) selectDocument(item.dataset.document);
    });
    $("examples").addEventListener("click", (event) => {
      const item = event.target.closest("[data-query]");
      if (item) {
        const example = currentDocument().examples[Number(item.dataset.query)];
        if (example) {
          $("question").value = example.query;
          execute();
        }
      }
    });
    for (const id of ["page-tabs", "rankings", "citation-list"])
      $(id).addEventListener("click", (event) => {
        const item = event.target.closest("[data-page]");
        if (item) goToPage(Number(item.dataset.page), true);
      });
    $("query-form").addEventListener("submit", (event) => {
      event.preventDefault();
      execute();
    });
    $("question").addEventListener("input", () => {
      $("query-count").textContent = `${$("question").value.length} / 500`;
      markStale();
    });
    $("question").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        execute();
      }
    });
    $("previous-page").addEventListener("click", () =>
      goToPage(state.page - 1),
    );
    $("next-page").addEventListener("click", () => goToPage(state.page + 1));
    $("unavailable-button").addEventListener("click", () => {
      const page = currentDocument().pages.find((item) => item.status !== "ok");
      if (page) goToPage(page.number, true);
    });
    $("top-k").addEventListener("change", execute);
    $("support-threshold").addEventListener("input", () => {
      $("threshold-value").textContent =
        `${Math.round(Number($("support-threshold").value) * 100)}%`;
      markStale();
    });
    $("support-threshold").addEventListener("change", execute);
    $("validate-response").addEventListener("click", validateEditor);
    $("reset-response").addEventListener("click", restoreEditor);
    $("bad-citation").addEventListener("click", () => {
      if (state.result) {
        $("response-json").value = JSON.stringify(
          {
            ...state.result.response,
            answer:
              state.result.response.answer || "Edited demonstration response",
            abstain: false,
            evidence_pages: [99],
          },
          null,
          2,
        );
        validateEditor();
      }
    });
    $("unsupported-answer").addEventListener("click", () => {
      if (state.result) {
        $("response-json").value = JSON.stringify(
          {
            ...state.result.response,
            answer: "This sentence does not occur in the supplied pages.",
            abstain: false,
            evidence_pages: state.result.retrieval.supplied_pages.slice(0, 1),
          },
          null,
          2,
        );
        validateEditor();
      }
    });
    $("export-result").addEventListener("click", exportResult);
    selectDocument(state.documentId);
    return {
      state,
      execute,
      goToPage,
      selectDocument,
      validateEditor,
      exportResult,
    };
  }
  root.LavaApp = { mount };
  if (typeof module !== "undefined" && module.exports)
    module.exports = { mount };
  if (root.document) {
    try {
      root.lavaDemo = mount(root.document);
    } catch (error) {
      const target = root.document.getElementById("run-status");
      if (target)
        target.textContent = "Workspace unavailable: " + error.message;
    }
  }
})(globalThis);
