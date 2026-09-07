"""Accessible public retrieval tables and exact SVG comparison charts."""

from __future__ import annotations

import html
from typing import Any

from lava.evaluation.walkthrough import TABLE_STYLE, render_table

LABELS = {"page_order": "Page-order control", "bm25": "Multilingual BM25"}
METRICS = {
    "recall_at_k": "Evidence recall",
    "all_evidence_at_k": "All evidence found",
    "reciprocal_rank_at_k": "MRR",
    "average_precision_at_k": "MAP",
    "ndcg_at_k": "nDCG",
}


def metric_rows(summary: dict[str, Any], weighting: str = "question_average") -> list[dict]:
    return [
        {
            "Method": LABELS[method],
            "Pages (k)": int(k),
            **{
                label: (
                    f"{values[key]:.1%}"
                    if key in {"recall_at_k", "all_evidence_at_k"}
                    else f"{values[key]:.3f}"
                )
                for key, label in METRICS.items()
            },
        }
        for method in LABELS
        for k, values in sorted(
            summary["methods"][method][weighting].items(), key=lambda pair: int(pair[0])
        )
    ]


def coverage_rows(summary: dict[str, Any]) -> list[dict]:
    return [
        {
            "PDF": row["document"],
            "All pages": row["page_count"],
            "Textless pages": row["textless_pages"],
            "Extraction errors": row["extraction_errors"],
            "Gold pages without text": row["gold_pages_without_text"],
        }
        for row in summary["document_coverage"]
    ]


def recall_chart(summary: dict[str, Any], metric: str = "recall_at_k") -> str:
    """Render two exact bar series at fixed budgets, with explicit 0–100% axes."""
    budgets = summary["implementation"]["config"]["budgets"]
    title = METRICS[metric]
    elements = [
        f'<svg viewBox="0 0 840 330" role="img" aria-label="{html.escape(title)} by page budget" style="width:100%;max-width:840px;font-family:system-ui;background:#fff">',
        f"<title>{html.escape(title)} by page budget</title>",
        f'<text x="56" y="28" font-size="20" font-weight="700" fill="#13233f">{html.escape(title)}</text>',
    ]
    for percent in (0, 25, 50, 75, 100):
        y = 254 - 1.8 * percent
        elements.extend(
            [
                f'<line x1="60" y1="{y}" x2="820" y2="{y}" stroke="#e4e9f0"/>',
                f'<text x="52" y="{y + 4}" text-anchor="end" font-size="12" fill="#526077">{percent}%</text>',
            ]
        )
    group_width = 740 / len(budgets)
    for i, k in enumerate(budgets):
        for j, (method, color) in enumerate((("page_order", "#a9b4c5"), ("bm25", "#176f83"))):
            value = summary["methods"][method]["question_average"][str(k)][metric]
            x, height = 78 + i * group_width + j * 48, 180 * value
            elements.extend(
                [
                    f'<rect x="{x}" y="{254 - height}" width="38" height="{height}" rx="3" fill="{color}"><title>{LABELS[method]}, k={k}: {value:.1%}</title></rect>',
                    f'<text x="{x + 19}" y="{246 - height}" text-anchor="middle" font-size="11" fill="#13233f">{value:.0%}</text>',
                ]
            )
        elements.append(
            f'<text x="{121 + i * group_width}" y="278" text-anchor="middle" font-size="13" fill="#13233f">k = {k}</text>'
        )
    elements.extend(
        [
            '<rect x="64" y="307" width="12" height="12" fill="#a9b4c5"/>',
            '<text x="83" y="317" font-size="12" fill="#526077">Page-order control</text>',
            '<rect x="240" y="307" width="12" height="12" fill="#176f83"/>',
            '<text x="259" y="317" font-size="12" fill="#526077">Multilingual BM25</text>',
            "</svg>",
        ]
    )
    return "".join(elements)


def document_rows(summary: dict[str, Any]) -> list[dict]:
    budgets = summary["implementation"]["config"]["budgets"]
    return [
        {
            "PDF": document,
            **{
                f"All evidence @ {k}": f"{values[str(k)]['all_evidence_at_k']:.1%}" for k in budgets
            },
        }
        for document, values in summary["methods"]["bm25"]["by"]["document"].items()
    ]


def render_report(summary: dict[str, Any]) -> str:
    pages = sum(row["page_count"] for row in summary["document_coverage"])
    body = (
        '<header><p class="eyebrow">LAVA · RETRIEVAL EXPERIMENT</p><h1>Can we find the evidence?</h1>'
        f"<p>{summary['question_count']} questions · {summary['document_count']} PDFs · {pages} physical pages</p></header>"
        "<p>Frozen text-search baseline. Every page is eligible; gold answers and evidence labels are excluded from ranking. "
        "These are training diagnostics, not held-out or organizer-server scores.</p>"
        "<h2>Coverage before scoring</h2>"
        + render_table(coverage_rows(summary), caption="Complete PDF coverage")
        + "<h2>Evidence found at each page budget</h2>"
        + recall_chart(summary)
        + recall_chart(summary, "all_evidence_at_k")
        + render_table(metric_rows(summary), caption="Question-weighted retrieval metrics")
        + "<h2>Do results hold across PDFs?</h2>"
        + render_table(document_rows(summary), caption="Complete evidence by PDF · BM25")
        + render_table(
            metric_rows(summary, "document_average"), caption="Equal-document retrieval metrics"
        )
        + "<h2>What the result does—and does not—measure</h2><p>Recall counts how much evidence was found. "
        "Complete-evidence coverage requires every gold page. MRR rewards an early first hit; MAP and nDCG reward a useful ordering. "
        "MAP@k uses min(number of relevant pages, k) as its denominator. Budgets were fixed before scoring.</p>"
        "<p>The previous 9B reader score used oracle pages. A new answer score requires actually running the reader on these retrieved pages. "
        "Native text cannot read pixels in scans, charts or photographs. Textless and failed pages remain in coverage.</p>"
        f"<footer>Completed {html.escape(summary['completed_at_utc'])} · First CPU run {summary['first_run_seconds']:.2f}s · "
        f"Contract {html.escape(summary['contract_id'][:12])} · Alvaro Mendizabal</footer>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>LAVA · Evidence retrieval</title>'
        + TABLE_STYLE
        + "<style>body{font:16px/1.65 system-ui;margin:0;background:#edf2f6;color:#13233f}main{max-width:1060px;margin:36px auto;padding:36px;background:white;border-radius:16px}h1{font-size:38px;line-height:1.15}h2{margin-top:38px;font-size:23px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#176f83}footer{border-top:1px solid #dbe3eb;margin-top:32px;padding-top:18px;font-size:12px;color:#526077}@media(max-width:650px){main{margin:0;padding:18px}h1{font-size:30px}}</style>"
        "</head><body><main>" + body + "</main></body></html>"
    )
