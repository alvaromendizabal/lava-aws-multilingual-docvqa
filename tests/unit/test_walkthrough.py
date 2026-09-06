"""Notebook views must preserve measured coverage, score validity and cost scope."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from lava.evaluation.reporting import load_report
from lava.evaluation.walkthrough import comparison_tables, render_table, training_rates

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def report():
    return load_report(ROOT)


@pytest.fixture
def prices():
    return json.loads((ROOT / "reports/aws/training_prices.json").read_text())


def test_saved_quality_and_cost_are_calculated_from_verified_run(report, prices):
    tables = comparison_tables(report, training_rates(prices))
    nine = next(row for row in tables["quality"] if row["Reader"] == "Qwen3.5 · 9B")
    assert nine["Local LAVA overall"] == pytest.approx(0.8702380952380953)
    assert nine["Local LAVA overall"] == pytest.approx(
        (nine["Semantic VQA"] + nine["Evidence F1"]) / 2
    )
    cost = next(row for row in tables["systems"] if row["Reader"] == "Qwen3.5 · 9B")
    assert cost["Estimated compute (USD)"] == pytest.approx(395 * 2.8 / 3600)
    assert cost["Generation p95 (s)"] == pytest.approx(5.15651527125003)
    lineage = next(row for row in tables["lineage"] if row["Reader"] == "Qwen3.5 · 9B")
    assert lineage["Weights"] == "bfloat16"


def test_smoke_never_receives_comparable_quality_or_throughput(report, prices):
    smoke = next(run for run in report["runs"] if not run["complete"])
    # Even an accidentally attached semantic result cannot turn a smoke into a pilot.
    smoke = deepcopy(smoke)
    smoke["semantic_summary"] = deepcopy(
        next(run["semantic_summary"] for run in report["runs"] if run["semantic_summary"])
    )
    report["current_models"] = [smoke]
    tables = comparison_tables(report, training_rates(prices))
    assert tables["quality"][0]["Local LAVA overall"] is None
    assert tables["quality"][0]["Semantic VQA"] is None
    assert tables["systems"] == []
    assert "Smoke only" in tables["coverage"][0]["Stage"]


def test_stale_judge_and_unknown_prices_remain_unscored(report):
    report = deepcopy(report)
    for run in report["current_models"]:
        if run["semantic_summary"]:
            run["semantic_summary"]["contract_current"] = False
    tables = comparison_tables(report, {})
    assert all(row["Local LAVA overall"] is None for row in tables["quality"])
    assert all(row["Estimated compute (USD)"] is None for row in tables["systems"])


@pytest.mark.parametrize(
    "change", ["studio", "unit", "nan", "negative", "bool", "region", "duplicate"]
)
def test_pricing_rejects_wrong_service_scope_or_invalid_rates(prices, change):
    if change == "studio":
        prices["products"][0]["component"] = "studio-jupyterlab"
    elif change == "unit":
        prices["products"][0]["unit"] = "Seconds"
    elif change == "region":
        prices["region"] = "us-east-1"
    elif change == "duplicate":
        prices["products"].append(deepcopy(prices["products"][0]))
    else:
        prices["products"][0]["usd_per_hour"] = {"nan": float("nan"), "negative": -1, "bool": True}[
            change
        ]
    with pytest.raises((TypeError, ValueError)):
        training_rates(prices)


@pytest.mark.parametrize("value", [float("nan"), -1, True])
def test_invalid_billable_duration_cannot_produce_a_cost(report, value):
    report = deepcopy(report)
    next(run for run in report["current_models"] if run["complete"])["billable_seconds"] = value
    with pytest.raises(ValueError, match="Billable"):
        comparison_tables(report, {"ml.g5.2xlarge": 1.515})


def test_table_is_accessible_escaped_and_distinguishes_zero_from_missing():
    rendered = render_table(
        [{"Reader": "<script>bad</script>", "Score": 0.0, "Missing": None}],
        percent_columns=("Score",),
        caption="Scores & coverage",
    )
    assert '<th scope="col">Score</th>' in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "0.00%" in rendered
    assert "Not evaluated" in rendered
    assert "<caption>Scores &amp; coverage</caption>" in rendered
