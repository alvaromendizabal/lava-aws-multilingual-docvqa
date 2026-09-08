"""Generate static figures and a Plotly report from verified integrated results.

Run with `make system-report`. The isolated plotting environment deliberately
leaves the frozen semantic-judge environment and its dependency lock unchanged.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lava.readers.runtime_logging import RuntimeEventLogger


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    import matplotlib  # type: ignore[import-not-found,import-untyped]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found,import-untyped]

    folder = ROOT / "reports/system"
    source = folder / "summary.json"
    payload = source.read_bytes()
    if sha(payload) != source.with_suffix(".sha256").read_text().strip():
        raise ValueError("System summary checksum mismatch")
    report = json.loads(payload)
    rows = report["per_question"]
    if len(rows) != 16 or report["split"] != "training_diagnostic":
        raise ValueError("Figures require the complete measured training diagnostic")
    logger = RuntimeEventLogger("system.figures")
    logger.emit("figures.source.verified", questions=len(rows), summary_sha256=sha(payload))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "text.color": "#153047",
            "axes.labelcolor": "#153047",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "svg.fonttype": "none",
            "svg.hashsalt": "lava-system",
        }
    )
    colors = ("#9aabba", "#087e8b")
    metrics = ("answer", "grounding", "overall")
    labels = ("Answer credit", "Evidence F1", "Local LAVA")
    oracle = report["oracle_question_micro"]
    measured = report["metrics"]["question_micro"]
    traces = []
    with logger.stage("figures.render", heartbeat_seconds=15):
        fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
        for index, (name, values) in enumerate(
            (("Oracle pages", oracle), ("Retrieved pages", measured))
        ):
            values_pct = [100 * values[metric] for metric in metrics]
            bars = ax.bar(
                [x + (index - 0.5) * 0.32 for x in range(3)],
                values_pct,
                width=0.30,
                color=colors[index],
                label=name,
            )
            ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values_pct], padding=4)
            traces.append(
                {
                    "type": "bar",
                    "name": name,
                    "x": labels,
                    "y": values_pct,
                    "marker": {"color": colors[index]},
                    "hovertemplate": "%{x}: %{y:.2f}%<extra>%{fullData.name}</extra>",
                }
            )
        ax.set(
            xticks=range(3), xticklabels=labels, ylim=(0, 108), ylabel="Question-average score (%)"
        )
        ax.set_yticks(range(0, 101, 25))
        ax.set_title(
            "From supplied evidence to retrieved evidence", loc="left", fontweight="bold", pad=20
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2, frameon=False)
        ax.yaxis.grid(True, color="#e7edf1")
        ax.set_axisbelow(True)
        fig.savefig(folder / "quality.svg", metadata={"Date": None}, facecolor="white")
        plt.close(fig)

        docs = sorted(report["metrics"]["by_document"])
        paired = report["paired_document_comparison"]
        # Oracle document values are read from the same checksum-bound comparison source.
        baseline_path = (
            ROOT
            / "reports/oracle_reader/runs"
            / report["contract"]["config"]["oracle_job"]
            / "semantic_summary.json"
        )
        baseline_bytes = baseline_path.read_bytes()
        if sha(baseline_bytes) != report["oracle_summary_sha256"]:
            raise ValueError("Document comparison oracle source mismatch")
        baseline = json.loads(baseline_bytes)["metrics"]["by_document"]
        fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
        for index, (name, values) in enumerate(
            (("Oracle pages", baseline), ("Retrieved pages", report["metrics"]["by_document"]))
        ):
            ax.bar(
                [x + (index - 0.5) * 0.32 for x in range(len(docs))],
                [100 * values[doc]["overall"] for doc in docs],
                width=0.30,
                color=colors[index],
                label=name,
            )
        ax.set(xticks=range(len(docs)), xticklabels=docs, ylim=(0, 105), ylabel="Local LAVA (%)")
        ax.set_yticks(range(0, 101, 25))
        ax.set_title(
            "How does performance vary across documents?", loc="left", fontweight="bold", pad=16
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2, frameon=False)
        ax.yaxis.grid(True, color="#e7edf1")
        ax.set_axisbelow(True)
        fig.savefig(folder / "documents.svg", metadata={"Date": None}, facecolor="white")
        plt.close(fig)

        matrix = [
            [row[key] for key in ("answer_score", "evidence_f1", "local_lava")] for row in rows
        ]
        fig, ax = plt.subplots(figsize=(9, 6.7), layout="constrained")
        ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(
            xticks=range(3),
            xticklabels=labels,
            yticks=range(len(rows)),
            yticklabels=[f"{r['question']} · {r['document']}" for r in rows],
        )
        ax.tick_params(length=0)
        for i, values in enumerate(matrix):
            for j, value in enumerate(values):
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#153047",
                    fontsize=9,
                )
        ax.set_title(
            "Every question remains in the evaluation", loc="left", fontweight="bold", pad=18
        )
        fig.savefig(folder / "questions.svg", metadata={"Date": None}, facecolor="white")
        plt.close(fig)

        escaped = html.escape
        table = "".join(
            "<tr>"
            + "".join(
                f"<td>{escaped(str(value))}</td>"
                for value in (
                    row["question"],
                    row["document"],
                    f"{row['answer_score']:.1%}",
                    f"{row['evidence_f1']:.1%}",
                    row["failure_category"].replace("_", " "),
                )
            )
            + "</tr>"
            for row in rows
        )
        plots = json.dumps(traces).replace("<", "\\u003c")
        heatmap = json.dumps(
            {
                "type": "heatmap",
                "z": matrix,
                "x": labels,
                "y": [f"{r['question']} · {r['document']}" for r in rows],
                "zmin": 0,
                "zmax": 1,
                "colorscale": "YlGnBu",
                "hovertemplate": "%{y}<br>%{x}: %{z:.1%}<extra></extra>",
            }
        )
        svg = {
            name: (folder / name).read_text().split("<svg", 1)[1]
            for name in ("quality.svg", "documents.svg", "questions.svg")
        }
        body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LAVA | Measured document intelligence</title>
<style>body{{margin:0;background:#f0f4f6;color:#153047;font:16px/1.65 system-ui}}main{{max-width:1080px;margin:auto;padding:44px 24px}}header{{padding:24px 0 32px;border-bottom:1px solid #cad6de}}.eyebrow{{letter-spacing:.13em;font-size:12px;color:#087e8b;font-weight:700}}h1{{font-size:clamp(32px,5vw,54px);line-height:1.08;max-width:800px}}h2{{font-size:25px;margin-top:0}}.lede{{max-width:780px;font-size:19px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.card,section{{background:white;border-radius:12px;padding:24px}}.card strong{{display:block;font-size:32px;color:#087e8b}}section{{margin:20px 0}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px 8px;border-bottom:1px solid #e2e8ed;text-align:left}}.scroll{{overflow:auto}}.note,footer{{font-size:13px;color:#536777}}a{{color:#087e8b}}nav a{{margin-right:20px}}.interactive{{display:none;min-height:390px}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:16px}}}}</style></head><body><main>
<header><p class="eyebrow">ALVARO MENDIZABAL · APPLIED ML ENGINEERING</p><h1>Read the PDF.<br>Find the evidence.<br>Measure the answer.</h1><p class="lede">A reproducible document-AI study on AWS: multilingual retrieval, an open 9B vision-language reader, and an evaluation that keeps every failure visible.</p><nav><a href="#quality">Results</a><a href="#errors">Failure analysis</a><a href="#method">Method</a><a href="https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa">Repository</a></nav></header>
<div class="cards"><div class="card"><strong>{measured["overall"]:.2%}</strong>Retrieved-page local LAVA</div><div class="card"><strong>{report["diagnostics"]["schema_valid_rate"]:.0%}</strong>Valid structured responses</div><div class="card"><strong>16 / 5</strong>Labeled questions / PDFs</div></div>
<p class="note">Previously examined training data: 15 Japanese questions and one Vietnamese question. Local implementation of the published LAVA formula; organizer-server parity and held-out performance are not established.</p>
<section id="quality"><h2>What changes when the system must retrieve its own evidence?</h2><p>The same pinned reader and semantic judge evaluate both conditions. The retrieved-page score changes by <strong>{100 * report["question_mean_delta"]:+.2f} percentage points</strong> relative to supplied gold pages. Answers and citations are scored separately.</p><div id="quality-plot" class="interactive"></div><div id="quality-static"><svg{svg["quality.svg"]}</div><p class="note">Both charts use 0–100% score scales. The oracle condition supplies the correct evidence pages; the integrated condition uses five BM25-selected pages.</p><svg{svg["documents.svg"]}</section>
<section id="errors"><h2>Inspect the failures that the average hides</h2><p>Retrieval misses, answer errors and citation errors require different remedies. These categories guide inspection; they do not establish a causal explanation.</p><div id="question-plot" class="interactive"></div><div id="question-static"><svg{svg["questions.svg"]}</div><div class="scroll"><table><thead><tr><th>Question</th><th>Document</th><th>Answer</th><th>Evidence</th><th>Diagnostic</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section id="method"><h2>Why this is reproducible</h2><p>Immutable model and data revisions, deterministic decoding, label-free retrieval inputs, per-question S3 checkpoints, independent parsing, and a pinned judge bind the result to the implementation. Repeated work reuses validated checkpoints.</p><p>The 1,582-candidate lexical audit records training-only selection in every document fold. A separate visual retriever provides a negative result and an exploratory hybrid. Neither automatically changes the production baseline.</p><details><summary>Metric, runtime and provenance</summary><p>The local LAVA score averages semantic answer credit and evidence-page F1. The judge passes 28 public development controls. Five documents are insufficient to establish generalization or state-of-the-art performance.</p><p>Mean generation: {report["runtime"]["generation_mean_seconds"]:.2f}s; p95: {report["runtime"]["generation_p95_seconds"]:.2f}s; peak allocated GPU memory: {report["runtime"]["peak_allocated_gib"]:.2f} GiB. These measurements exclude provisioning, retrieval and checkpoint I/O.</p><p>System contract: <code>{escaped(report["contract"]["contract_id"])}</code><br>Summary SHA-256: <code>{sha(payload)}</code></p><p>Paired document diagnostics: {escaped(json.dumps(paired, sort_keys=True))}</p></details></section>
<footer>All values come from checksum-verified public experiment artifacts. Charts use Matplotlib {escaped(matplotlib.__version__)} with optional Plotly.js 4.0.0 interaction. Static figures remain available if the Plotly CDN cannot load. <a href="https://lava-workshop.github.io/#evaluation">Published metric</a> · <a href="https://plotly.com/javascript/getting-started/">Plotly documentation</a></footer>
</main><script src="https://cdn.plot.ly/plotly-4.0.0.min.js"></script><script>
if(window.Plotly){{const base={{font:{{family:'system-ui',color:'#153047'}},paper_bgcolor:'white',plot_bgcolor:'white',margin:{{t:30,b:70,l:70,r:20}}}};
document.getElementById('quality-plot').style.display='block';
Plotly.newPlot('quality-plot',{plots},{{...base,barmode:'group',yaxis:{{range:[0,105],ticksuffix:'%'}},legend:{{orientation:'h',y:-0.2}}}},{{responsive:true,displaylogo:false}}).then(()=>document.getElementById('quality-static').style.display='none').catch(()=>document.getElementById('quality-plot').style.display='none');
document.getElementById('question-plot').style.display='block';
Plotly.newPlot('question-plot',[{heatmap}],{{...base,height:650,margin:{{t:20,b:60,l:140,r:70}},yaxis:{{autorange:'reversed'}}}},{{responsive:true,displaylogo:false}}).then(()=>document.getElementById('question-static').style.display='none').catch(()=>document.getElementById('question-plot').style.display='none');}}
</script></body></html>"""
        (folder / "index.html").write_text(body)
        manifest = {
            "summary_sha256": sha(payload),
            "source_sha256": sha(Path(__file__).read_bytes()),
            "matplotlib_version": matplotlib.__version__,
            "plotly_js_version": "4.0.0",
            "files": {name: sha((folder / name).read_bytes()) for name in (*svg, "index.html")},
        }
        (folder / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.emit("figures.completed", static_figures=3, interactive_report=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
