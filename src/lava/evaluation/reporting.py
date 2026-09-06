"""Reproducible public reader reports built from verified aggregate artifacts."""

from __future__ import annotations

import hashlib
import html
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from lava.evaluation.statistics import compare_document_scores
from lava.readers.evaluation_contract import load_evaluation_contract

LABELS = {
    "qwen35_4b_fused_direct": "Qwen3.5 · 4B",
    "qwen35_9b_fused_direct": "Qwen3.5 · 9B",
    "qwen38_27b_nf4_g5_fused_direct": "Qwen3.8 · 27B NF4",
}


def load_report(root: Path) -> dict[str, Any]:
    """Verify local summary hashes and distinguish smoke runs from complete pilots."""
    contract = load_evaluation_contract(root)
    runs = []
    for path in sorted((root / "reports/oracle_reader/runs").glob("*/sync_manifest.json")):
        manifest = json.loads(path.read_text())
        payload = (path.parent / "public_summary.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["public_summary_sha256"]:
            raise ValueError(f"Public summary checksum mismatch: {path.parent.name}")
        summary = json.loads(payload)
        gate = manifest["artifact_gate"]
        for key in (
            "model_key",
            "model_id",
            "model_revision",
            "git_commit_sha",
            "protocol_lock_id",
        ):
            if manifest[key] != summary[key]:
                raise ValueError(f"Public manifest lineage mismatch: {key}")
        if gate["raw_response_count"] != summary["record_count"]:
            raise ValueError("Artifact gate and summary counts differ")
        if summary["protocol_lock_id"] != contract["protocol_lock_id"]:
            raise ValueError("Run protocol differs from the report contract")
        if summary["asset_manifest_sha256"] != contract["private_manifest_sha256"]:
            raise ValueError("Run manifest differs from the report contract")
        complete = summary.get("run_kind") == "benchmark"
        if complete:
            required = {
                "record_count": contract["question_count"],
                "document_count": contract["document_count"],
                "document_question_counts": contract["document_question_counts"],
                "language_counts": contract["language_counts"],
                "answer_format_counts": contract["answer_format_counts"],
                "coverage_status": "complete_frozen_pilot",
            }
            if any(summary.get(k) != v for k, v in required.items()):
                raise ValueError("Incomplete benchmark cannot enter full-pilot comparisons")
        for key in (
            "normalized_exact_answer_micro",
            "normalized_exact_answer_document_macro",
            "schema_valid_rate",
            "self_grounding_f1_micro",
            "abstention_rate",
        ):
            value = summary[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid score: {key}")
        for key in ("mean_generation_seconds", "max_peak_cuda_memory_allocated_mib"):
            if not math.isfinite(summary[key]) or summary[key] < 0:
                raise ValueError(f"Invalid telemetry: {key}")
        runs.append(
            {
                "job_name": manifest["job_name"],
                "label": LABELS.get(summary["model_key"], summary["model_key"]),
                "model_key": summary["model_key"],
                "instance_type": manifest["instance_type"],
                "billable_seconds": manifest["billable_time_seconds"],
                "summary_sha256": manifest["public_summary_sha256"],
                "complete": complete,
                "summary": summary,
            }
        )
    comparisons = []
    for left, right in combinations([r for r in runs if r["complete"]], 2):
        a, b = left["summary"], right["summary"]
        # Vary the reader; hold the evaluated questions and prompting fixed.
        keys = (
            "protocol_lock_id",
            "asset_manifest_sha256",
            "prompt_version",
            "input_mode",
            "generation",
            "metric_boundary",
            "document_question_counts",
        )
        if any(a.get(key) != b.get(key) for key in keys):
            continue
        comparisons.append(
            {
                "baseline_job": left["job_name"],
                "challenger_job": right["job_name"],
                **compare_document_scores(
                    a["document_scores"], b["document_scores"], seed=20260902
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_lock_id": contract["protocol_lock_id"],
        "expected_questions": contract["question_count"],
        "expected_documents": contract["document_count"],
        "runs": runs,
        "paired_document_comparisons": comparisons,
        "interpretation": "Descriptive normalized-exact pilot; semantic judging and model promotion pending.",
    }


def _chart(runs: list[dict[str, Any]], key: str, title: str, unit: str, divisor: float = 1) -> str:
    values = [float(r["summary"][key]) / divisor for r in runs]
    maximum = max(values, default=1) or 1
    bars = []
    for index, (run, value) in enumerate(zip(runs, values, strict=True)):
        y = 28 + index * 58
        width = 330 * value / maximum
        label = html.escape(run["label"])
        bars.append(
            f'<text x="0" y="{y}" class="label">{label}</text>'
            f'<rect x="0" y="{y + 10}" width="{width:.2f}" height="10" rx="5" fill="#147d92"/>'
            f'<text x="{width + 10:.2f}" y="{y + 20}" class="value">{value:.2f} {unit}</text>'
        )
    height = 30 + len(runs) * 58
    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 455 {height}">'
        + "".join(bars)
        + "</svg></section>"
    )


def render_report(report: dict[str, Any]) -> str:
    """Render an offline, self-contained report with escaped public fields only."""
    runs = report["runs"]
    complete = sum(r["complete"] for r in runs)
    headline = (
        "Reader execution is verified.<br>Full evaluation comes next."
        if not complete
        else "Reader evaluation.<br>Evidence before model selection."
    )
    rows = []
    for run in runs:
        s = run["summary"]
        values = [
            run["label"],
            run["instance_type"],
            "Full pilot" if run["complete"] else "Smoke only",
            str(s["record_count"]),
            f"{s['normalized_exact_answer_micro']:.3f}",
            f"{s['schema_valid_rate']:.1%}",
            f"{s['abstention_rate']:.1%}",
            str(run["billable_seconds"]),
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    headers = [
        "Reader",
        "GPU instance",
        "Coverage",
        "Questions",
        "Exact diagnostic",
        "Valid output",
        "Abstain",
        "Billable s",
    ]
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    panels = _chart(runs, "mean_generation_seconds", "Generation time per question", "s")
    panels += _chart(
        runs, "max_peak_cuda_memory_allocated_mib", "Peak allocated GPU memory", "GiB", 1024
    )
    detail = []
    for run in runs:
        s = run["summary"]
        detail.append(
            f"<details><summary>{html.escape(run['label'])} · lineage and slices</summary>"
            f"<p>{html.escape(run['job_name'])}</p><pre>"
            + html.escape(
                json.dumps(
                    {
                        "model_revision": s["model_revision"],
                        "code_revision": s["git_commit_sha"],
                        "summary_sha256": run["summary_sha256"],
                        "document_scores": s["document_scores"],
                        "by_language": s.get(
                            "by_language", {"coverage_only": s["language_counts"]}
                        ),
                        "by_answer_format": s.get(
                            "by_answer_format", {"coverage_only": s["answer_format_counts"]}
                        ),
                        "parser_errors": s["parser_error_counts"],
                    },
                    indent=2,
                )
            )
            + "</pre></details>"
        )
    comparison_html = (
        "<p>Pending: at least two complete, compatible 16-question runs are required.</p>"
    )
    if report["paired_document_comparisons"]:
        comparison_html = (
            "<pre>"
            + html.escape(json.dumps(report["paired_document_comparisons"], indent=2))
            + "</pre>"
        )
    rendered = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LAVA | Reader evaluation</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#192e40;font:15px/1.65 system-ui,sans-serif}
main{max-width:1240px;margin:auto;padding:52px 32px}.eyebrow{letter-spacing:.19em;font-size:12px;font-weight:700;color:#147d92}
h1{font-size:48px;letter-spacing:-.045em;line-height:1.13;margin:16px 0}h2{font-size:19px;line-height:1.4;margin:0 0 18px}
p{max-width:920px}.lead{font-size:18px;color:#506477}.cards,.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:30px 0}
.card,.panel{background:white;border:1px solid #dce4e9;border-radius:14px;padding:24px}.card strong{font-size:34px;display:block;line-height:1.3}
.card span{color:#506477}.charts{grid-template-columns:1fr 1fr}.notice{border-left:4px solid #d9a344;background:#fff8e9;padding:17px 22px;border-radius:0 9px 9px 0}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;min-width:950px;width:100%;font-size:13px}th{text-align:left;color:#607181;font-weight:600}td,th{padding:14px 12px;border-bottom:1px solid #e2e8ed}td:nth-child(5){font-variant-numeric:tabular-nums}
svg{width:100%;display:block}.label{font:13px system-ui;fill:#243f50}.value{font:12px system-ui;fill:#506477}
pre{overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f4f6f8;padding:18px;border-radius:8px}
details{border-top:1px solid #dce4e9;padding:18px 0}summary{cursor:pointer;font-weight:600}.muted,footer{color:#607181;font-size:13px}footer{margin-top:28px}
@media(max-width:700px){main{padding:28px 18px}h1{font-size:35px}.cards,.charts{grid-template-columns:1fr}.panel{padding:18px}}
</style></head><body><main><div class="eyebrow">LAVA / DOCUMENT INTELLIGENCE / AWS</div>
<h1>HEADLINE_PLACEHOLDER</h1>
<p class="lead">A traceable comparison of open vision-language readers with the correct evidence pages held fixed.</p>
"""
        + f"""<div class="cards"><div class="card"><strong>{len(runs)}</strong><span>Verified runs in this report</span></div>
<div class="card"><strong>{complete}</strong><span>Complete 16-question pilots</span></div>
<div class="card"><strong>16 / 5</strong><span>Frozen questions / documents</span></div></div>
<div class="notice"><strong>Read the coverage before the score.</strong> One-question smoke tests verify execution.
They cannot establish a winning reader. Exact matching is a diagnostic; the pinned semantic judge remains a separate gate.</div>
<div class="charts">{panels}</div>
<section class="panel"><h2>Observed results · every run retained</h2><div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Generation time excludes model loading. Billable seconds include the training job's measured billable duration; they are not dollar costs. Hardware and quantization differ across readers.</p></section>
<section class="panel" style="margin-top:24px"><h2>Document-level comparisons</h2>{comparison_html}
<p class="muted">Bootstrap intervals are exploratory with five documents. The smallest two-sided exact sign-flip p-value with five nonzero document deltas is 0.0625. No model promotion is supported by this pilot alone.</p></section>
<section class="panel" style="margin-top:24px"><h2>Coverage and interpretation</h2><p>The frozen set has 15 Japanese questions and one Vietnamese question. Language is confounded with document identity. The Vietnamese result is a single example, not a language-level estimate.</p>
<p>These are frozen-model descriptive evaluations. The existence of nested fold manifests does not mean nested training, tuning, or cross-validation has been performed. Retrieval is held at oracle evidence; overall oracle diagnostics contain a fixed grounding contribution.</p></section>
<section class="panel" style="margin-top:24px"><h2>Audit trail</h2>{"".join(detail)}</section>
<footer>Alvaro Mendizabal · LAVA · Rebuild with make report · Protocol {html.escape(report["protocol_lock_id"][:16])}</footer></main></body></html>"""
    )
    return rendered.replace("HEADLINE_PLACEHOLDER", headline)


def write_report(root: Path) -> Path:
    """Write deterministic aggregate JSON and a self-contained public HTML report."""
    report = load_report(root)
    target = root / "reports/oracle_reader/evaluation"
    target.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "summary.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
        "index.html": render_report(report),
    }.items():
        path = target / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return target / "index.html"
