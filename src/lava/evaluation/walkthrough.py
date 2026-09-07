"""Offline, presentation-ready views of verified reader experiments."""

from __future__ import annotations

import html
import math
from typing import Any

from lava.evaluation.analysis import compare_semantic_runs, validate_semantic_metrics


def training_rates(snapshot: dict[str, Any]) -> dict[str, float]:
    """Reject prices for Studio, another region, or non-hourly/non-finite rates."""
    if snapshot["service_code"] != "AmazonSageMaker" or snapshot["region"] != "us-west-2":
        raise ValueError("Expected SageMaker pricing in the experiment region")
    rates = {}
    for product in snapshot["products"]:
        if product["component"] != "Training" or product["unit"] != "Hrs":
            raise ValueError("Reader costs require hourly Training prices, not Studio prices")
        rate = product["usd_per_hour"]
        if isinstance(rate, bool) or not isinstance(rate, (float, int)):
            raise TypeError("Training price must be numeric")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Training price must be finite and positive")
        instance = product["instance_type"]
        if instance in rates:
            raise ValueError("Ambiguous training price for an instance")
        rates[instance] = float(rate)
    return rates


def comparison_tables(report: dict[str, Any], rates: dict[str, float]) -> dict[str, Any]:
    """Keep coverage, semantic quality, measured resources and lineage distinct."""
    coverage, quality, systems, lineage = [], [], [], []
    for run in report["current_models"]:
        summary = run["summary"]
        semantic = run.get("semantic_summary")
        scored = bool(run["complete"] and semantic and semantic["contract_current"])
        status = (
            "Full pilot scored"
            if scored
            else "Full pilot; semantic scoring required"
            if run["complete"]
            else "Smoke only; full pilot required"
        )
        coverage.append(
            {
                "Reader": run["label"],
                "Questions": f"{summary['record_count']} / {report['expected_questions']}",
                "Stage": status,
            }
        )
        if scored:
            validate_semantic_metrics(semantic["metrics"], summary)
        metrics = semantic["metrics"]["question_micro"] if scored else {}
        quality.append(
            {
                "Reader": run["label"],
                "Semantic VQA": metrics.get("answer"),
                "Evidence F1": summary["self_grounding_f1_micro"] if run["complete"] else None,
                "Local LAVA overall": metrics.get("overall"),
                "Valid output": summary["schema_valid_rate"] if run["complete"] else None,
                "Abstention": summary["abstention_rate"] if run["complete"] else None,
            }
        )
        # A one-question setup test is not a comparable throughput or cost measurement.
        if run["complete"]:
            seconds = run["billable_seconds"]
            if isinstance(seconds, bool) or not isinstance(seconds, (float, int)):
                raise ValueError("Billable duration must be numeric")
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Billable duration must be finite and non-negative")
            rate = rates.get(run["instance_type"])
            if rate is not None and (not math.isfinite(rate) or rate <= 0):
                raise ValueError("Training price must be finite and positive")
            latency = (run.get("supporting_metrics") or {}).get("latency", {})
            systems.append(
                {
                    "Reader": run["label"],
                    "Instance": run["instance_type"],
                    "Generation mean (s)": summary["mean_generation_seconds"],
                    "Generation p95 (s)": latency.get("generation_p95_seconds"),
                    "Peak allocated (GiB)": summary["max_peak_cuda_memory_allocated_mib"] / 1024,
                    "Billable time (s)": seconds,
                    "Estimated compute (USD)": seconds * rate / 3600 if rate else None,
                }
            )
        lineage.append(
            {
                "Reader": run["label"],
                "Model": summary["model_id"],
                "Weights": summary["quantization"]
                if summary.get("quantization") not in (None, "none")
                else summary.get("dtype", "Not recorded"),
                "Model revision": summary["model_revision"],
                "Code revision": summary["git_commit_sha"],
                "Job": run["job_name"],
            }
        )
    return {"coverage": coverage, "quality": quality, "systems": systems, "lineage": lineage}


def candidate_assessment(
    report: dict[str, Any], baseline_key: str, challenger_key: str
) -> dict[str, Any]:
    """Describe a compatible full-pilot comparison without promoting a model."""
    runs = {run["model_key"]: run for run in report["current_models"]}
    baseline, challenger = runs.get(baseline_key), runs.get(challenger_key)
    if not baseline or not challenger or not baseline["complete"] or not challenger["complete"]:
        return {"Status": "Two complete pilots are required"}
    comparisons = compare_semantic_runs(baseline, challenger)
    if not comparisons:
        return {"Status": "Compatible current semantic scores are required"}
    overall = next(pair for pair in comparisons if pair["metric"] == "overall")
    delta = overall["question_mean_delta"]
    leader = (
        "Tied"
        if math.isclose(delta, 0, abs_tol=1e-12)
        else challenger["label"]
        if delta > 0
        else baseline["label"]
    )
    return {
        "Comparison": f"{challenger['label']} minus {baseline['label']}",
        "Higher question-average local score": leader,
        "Question delta (pp)": 100 * delta,
        "Document delta (pp)": 100 * overall["mean_delta"],
        "Challenger PDFs improved / tied / regressed": (
            f"{overall['documents_improved']} / {overall['documents_tied']} / "
            f"{overall['documents_regressed']}"
        ),
        "Interpretation": "Descriptive pilot; no held-out superiority or model promotion established",
    }


def render_table(
    rows: list[dict[str, Any]], *, percent_columns: tuple[str, ...] = (), caption: str
) -> str:
    """Render accessible, escaped tables; unavailable measurements remain unscored."""
    if not rows:
        return f"<p>{html.escape(caption)}: no eligible results yet.</p>"
    columns = list(rows[0])
    header = "".join(f'<th scope="col">{html.escape(k)}</th>' for k in columns)
    body = []
    for row in rows:
        if list(row) != columns:
            raise ValueError("Table columns must agree")
        cells = []
        for key, value in row.items():
            if value is None:
                text = "Not evaluated"
            elif isinstance(value, float):
                text = f"{value:.2%}" if key in percent_columns else f"{value:.3f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="lava-table"><table>'
        f"<caption>{html.escape(caption)}</caption><thead><tr>{header}</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table></div>"
    )


TABLE_STYLE = """<style>
.lava-table{overflow-x:auto;margin:18px 0;border:1px solid #dce4e9;border-radius:12px;
background:#fff;color:#192e40;font:14px/1.6 system-ui,sans-serif}
.lava-table table{border-collapse:collapse;width:100%;text-align:left!important}
.lava-table caption{text-align:left;padding:18px 16px;font-size:16px;font-weight:650;
color:#147d92;caption-side:top}
.lava-table th{background:#eef5f7;font-weight:600;text-align:left!important}
.lava-table th,.lava-table td{padding:12px 16px;border-bottom:1px solid #e2e8ed;
font-variant-numeric:tabular-nums;vertical-align:top;overflow-wrap:anywhere}
.lava-table tbody tr:last-child td{border-bottom:0}
</style>"""
