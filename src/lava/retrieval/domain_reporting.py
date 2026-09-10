"""Public, checksum-bound figures for the document-feature ablation."""

from __future__ import annotations

import html
import json
from pathlib import Path

from lava.evaluation.semantic import digest
from lava.retrieval.research import BASELINE

LABELS = {
    BASELINE: "BM25 baseline",
    "add_body": "Add body text",
    "add_headings": "Add headings",
    "add_blocks": "Add local blocks",
    "add_tables": "Add table text",
    "add_numeric": "Add exact numbers",
    "add_coverage": "Add query coverage",
    "add_adjacent": "Add neighboring pages",
    "all_document_features": "All seven signals",
    "without_body": "All except body",
    "without_headings": "All except headings",
    "without_blocks": "All except blocks",
    "without_tables": "All except tables",
    "without_numeric": "All except numbers",
    "without_coverage": "All except coverage",
    "without_adjacent": "All except neighbors",
    "query_complement_top4_plus1": "Four BM25 + query complement",
}


def publish_figures(root: Path) -> str:
    """Generate a static notebook figure and an optional interactive HTML view."""
    path = root / "reports/retrieval/document_features.json"
    payload = path.read_bytes()
    if digest(payload) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Document-feature report checksum mismatch")
    report = json.loads(payload)
    metrics = report["pooled_diagnostics_not_selection"]
    if set(metrics) != set(LABELS) or report["question_count"] != 16:
        raise ValueError("Unexpected public ablation catalog or denominator")
    names = list(LABELS)
    counts = [round(metrics[n]["all_evidence_at_k"] * 16) for n in names]
    recall = [metrics[n]["recall_at_k"] * 100 for n in names]
    colors = [
        "#13867c" if n == "add_blocks" else "#b75d48" if c < 14 else "#48677f"
        for n, c in zip(names, counts, strict=True)
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 760" role="img" aria-label="Document feature ablations on sixteen training questions" style="font-family:system-ui;background:white">',
        '<rect width="1080" height="760" fill="white"/>',
        '<text x="28" y="38" font-size="24" font-weight="700" fill="#13233f">Document features: pooled gains versus validation</text>',
        '<text x="28" y="64" font-size="14" fill="#526077">Five training PDFs · 16 questions · five-page budget · no new model inference</text>',
        '<text x="315" y="94" font-size="13" fill="#526077">Questions with every required evidence page</text>',
        '<text x="980" y="94" text-anchor="middle" font-size="13" fill="#526077">Recall@5</text>',
    ]
    for tick in (0, 4, 8, 12, 16):
        x = 315 + tick * 34
        parts.extend(
            [
                f'<line x1="{x}" x2="{x}" y1="112" y2="676" stroke="#e5ebef"/>',
                f'<text x="{x}" y="696" text-anchor="middle" font-size="12" fill="#526077">{tick}</text>',
            ]
        )
    for i, (name, count, rec, color) in enumerate(zip(names, counts, recall, colors, strict=True)):
        y = 120 + i * 32
        parts.extend(
            [
                f'<text x="295" y="{y + 15}" text-anchor="end" font-size="13" fill="#13233f">{html.escape(LABELS[name])}</text>',
                f'<rect x="315" y="{y}" width="{count * 34}" height="22" rx="3" fill="{color}"/>',
                f'<text x="{325 + count * 34}" y="{y + 16}" font-size="13" fill="#13233f">{count}/16</text>',
                f'<text x="980" y="{y + 16}" text-anchor="middle" font-size="13" fill="#13233f">{rec:.2f}%</text>',
            ]
        )
    parts.extend(
        [
            '<text x="28" y="729" font-size="14" fill="#13233f">Conservative document-isolated selection retained BM25 in all five folds.</text>',
            '<text x="28" y="751" font-size="12" fill="#526077">The block gain is concentrated in one document. These are retrieval diagnostics, not answer accuracy.</text>',
            "</svg>",
        ]
    )
    svg = "\n".join(parts)
    path.with_suffix(".svg").write_text(svg)
    trace = {
        "type": "bar",
        "orientation": "h",
        "x": counts,
        "y": [LABELS[n] for n in names],
        "customdata": recall,
        "marker": {"color": colors},
        "hovertemplate": "%{y}<br>Complete evidence: %{x}/16<br>Recall@5: %{customdata:.2f}%<extra></extra>",
    }
    layout = {
        "height": 760,
        "margin": {"l": 240, "r": 35, "t": 20, "b": 60},
        "font": {"family": "system-ui", "color": "#13233f"},
        "xaxis": {
            "range": [0, 16.7],
            "title": {"text": "Questions with complete evidence"},
            "dtick": 4,
        },
        "yaxis": {"autorange": "reversed"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
    }
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LAVA document-feature research</title><style>body{{margin:0;background:#f1f5f8;color:#13233f;font:16px system-ui}}main{{max-width:1120px;margin:auto;padding:32px}}section{{background:white;border-radius:16px;padding:24px}}h1{{font-size:30px}}p{{line-height:1.6}}svg{{width:100%;height:auto}}a{{color:#176f83}}</style>
<main><h1>Document-feature research</h1><p>Seventeen fixed policies, seven feature families, five training PDFs. The block signal adds one complete question in pooled diagnostics; all five conservative document folds retain BM25.</p>
<section><div id="interactive"></div><div id="fallback">{svg}</div></section><p>These are 16-question development retrieval results, not official Kaggle scores or evidence of answer accuracy. Hover to inspect a policy. The static figure remains available if Plotly cannot load.</p><p>Source SHA-256: <code>{digest(payload)}</code></p></main>
<script src="https://cdn.plot.ly/plotly-4.0.0.min.js"></script><script>if(window.Plotly){{Plotly.newPlot('interactive',[{json.dumps(trace)}],{json.dumps(layout)},{{responsive:true,displaylogo:false}}).then(()=>{{document.getElementById('fallback').hidden=true;}});}}</script></html>"""
    path.with_suffix(".html").write_text(page)
    return svg
