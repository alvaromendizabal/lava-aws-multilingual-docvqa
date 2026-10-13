"""Build a source-linked employer review and execute bounded public checks.

Python standard library only. The curated receipt verifies the named files and
commands, not private training, model quality, or all repository dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "Public component checks and recorded evidence; not private model reproduction."


def require(condition: bool, message: str) -> None:
    """Keep gates active under python -O as well as normal execution."""
    if not condition:
        raise ValueError(message)


def local_file(root: Path, value: str) -> Path:
    """Accept regular, non-symlink files within the reviewed checkout only."""
    require(isinstance(value, str) and bool(value), "Empty source path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "Source path escapes checkout")
    require("\\" not in value and path.as_posix() == value, "Noncanonical source path")
    candidate = root / path
    require(
        not any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)),
        "Symlink source path",
    )
    require(candidate.resolve().is_relative_to(root.resolve()), "Source path escapes checkout")
    require(candidate.is_file(), f"Missing reviewed file: {value}")
    return candidate


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_project(root: Path) -> dict:
    project = json.loads(local_file(root, "review/project.json").read_text(encoding="utf-8"))
    require(project["schema_version"] == 1, "Unsupported review schema")
    require(
        re.fullmatch(r"alvaromendizabal/[A-Za-z0-9_.-]+", project["repository"]) is not None,
        "Invalid repository",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", project["reviewed_base_commit"]) is not None,
        "Invalid reviewed base",
    )
    require(
        bool(project["cards"]) and bool(project["commands"]),
        "Review must include evidence and executable checks",
    )
    ids = [item["id"] for item in project["commands"]]
    require(len(ids) == len(set(ids)), "Duplicate command identities")
    for command in project["commands"]:
        require(
            re.fullmatch(r"[a-z][a-z0-9-]*", command["id"]) is not None, "Unsafe command identity"
        )
        require(command["runtime"] in {"python", "node"}, "Unsupported command runtime")
        require(all(isinstance(arg, str) for arg in command.get("args", [])), "Invalid arguments")
        local_file(root, command["entry"])
    for card in project["cards"]:
        require(
            card["audience"] in {"ML engineering", "Data science", "Applied AI"},
            "Unknown review audience",
        )
        for kind in ("implementation", "tests", "evidence"):
            local_file(root, card[kind])
    for value in project["read_more"]:
        local_file(root, value["path"])
    dependencies = project.get("source_dependencies", [])
    require(isinstance(dependencies, list), "Invalid source dependencies")
    require(all(isinstance(name, str) for name in dependencies), "Invalid dependency path")
    require(len(dependencies) == len(set(dependencies)), "Duplicate source dependencies")
    for name in dependencies:
        local_file(root, name)
    return project


def source_hashes(root: Path, project: dict) -> dict[str, str]:
    """Fingerprint the explicit review surface, without collecting source contents."""
    paths = {"review/project.json", "review/run.py", "review/test_run.py", "review/template.html"}
    paths.update(command["entry"] for command in project["commands"])
    for card in project["cards"]:
        paths.update(card[kind] for kind in ("implementation", "tests", "evidence"))
    paths.update(item["path"] for item in project["read_more"])
    paths.update(project.get("source_dependencies", []))
    return {name: digest(local_file(root, name)) for name in sorted(paths)}


def page(project: dict, receipt: dict | None = None, root: Path = ROOT) -> str:
    """Render an offline brief; static briefs never claim test success."""
    esc = html.escape
    base = "https://github.com/" + project["repository"] + "/blob/main/"

    def link(path: str, label: str) -> str:
        return '<a href="' + base + quote(path, safe="/") + '">' + esc(label) + "</a>"

    cards = []
    for number, card in enumerate(project["cards"], 1):
        refs = " · ".join(
            link(card[kind], kind.capitalize()) for kind in ("implementation", "tests", "evidence")
        )
        cards.append(
            f'<article><p class="eyebrow">{number:02d} / {esc(card["audience"])}</p>'
            f"<h3>{esc(card['title'])}</h3><p>{esc(card['decision'])}</p>"
            f'<p class="boundary">{esc(card["boundary"])}</p><p>{refs}</p></article>'
        )
    metrics = "".join(
        f"<div><strong>{esc(item['value'])}</strong><span>{esc(item['label'])}</span></div>"
        for item in project["signals"]
    )
    status = "Inspection brief · run the checks to produce a receipt"
    checks = "<p>No checks were executed to render this static brief.</p>"
    if receipt is not None:
        status = "Selected public checks passed · see exact scope below"
        rows = "".join(
            f"<tr><td>{esc(check['id'])}</td><td>{esc(check['status'])}</td>"
            f"<td>{check['seconds']:.3f}</td></tr>"
            for check in receipt["checks"]
        )
        checks = (
            "<table><caption>Commands executed for this receipt</caption><thead><tr>"
            "<th>Check</th><th>Outcome</th><th>Seconds</th></tr></thead><tbody>"
            + rows
            + '</tbody></table><p><a href="receipt.json">Machine-readable receipt</a>'
            ' · <a href="manifest.json">Artifact checksums</a></p>'
        )
    more = " · ".join(link(item["path"], item["label"]) for item in project["read_more"])
    template = local_file(root, "review/template.html").read_text(encoding="utf-8")
    return Template(template).substitute(
        title=esc(project["title"]),
        discipline=esc(project["discipline"]),
        summary=esc(project["summary"]),
        status=esc(status),
        metrics=metrics,
        more=more,
        cards="".join(cards),
        checks=checks,
        scope=esc(project["scope"]),
        private_boundary=esc(project["private_boundary"]),
        instructions=link("review/README.md", "Detailed review instructions"),
        overview=link("README.md", "Project overview"),
        base_commit=esc(project["reviewed_base_commit"]),
        receipt_scope=esc(SCOPE),
    )


def verify(output: Path, root: Path, project: dict, with_node: bool) -> dict:
    manifest = json.loads(local_file(output, "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported artifact manifest")
    names = manifest["files"]
    require("receipt.json" in names and "index.html" in names, "Incomplete review bundle")
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and p.relative_to(output).as_posix() != "manifest.json"
    }
    require(set(names) == actual, "Unexpected or missing review artifacts")
    require(not any(p.is_symlink() for p in output.rglob("*")), "Symlink in review bundle")
    for name, expected in names.items():
        require(digest(local_file(output, name)) == expected, f"Changed review artifact: {name}")
    receipt = json.loads(local_file(output, "receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "passed" and receipt["scope"] == SCOPE, "Invalid review receipt")
    require(receipt["repository"] == project["repository"], "Repository identity mismatch")
    require(
        receipt["with_node"] is with_node, "Different check mode; use a separate output directory"
    )
    require(
        receipt["source_sha256"] == source_hashes(root, project),
        "Reviewed source changed; use a new output directory",
    )
    expected_ids = [c["id"] for c in project["commands"] if c["runtime"] == "python" or with_node]
    require([c["id"] for c in receipt["checks"]] == expected_ids, "Incomplete command coverage")
    require(
        all(c["status"] == "passed" and c["returncode"] == 0 for c in receipt["checks"]),
        "A command did not pass",
    )
    return receipt


def execute(root: Path, project: dict, output: Path, with_node: bool, timeout: int) -> dict:
    require(timeout > 0, "Timeout must be positive")
    require(
        not output.resolve().is_relative_to(root.resolve()),
        "Write review outputs outside the checkout",
    )
    if output.exists():
        receipt = verify(output, root, project, with_node)
        return {
            "status": "passed",
            "reused": True,
            "checks": len(receipt["checks"]),
            "output": str(output),
        }
    require(
        not with_node or shutil.which("node") is not None, "Node.js is required for --with-node"
    )
    before = source_hashes(root, project)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ml-review-", dir=output.parent))
    checks = []
    started = time.monotonic()
    try:
        for command in project["commands"]:
            if command["runtime"] == "node" and not with_node:
                continue
            args = [arg.replace("{output}", str(staging)) for arg in command.get("args", [])]
            entry = str(local_file(root, command["entry"]))
            argv = (
                [sys.executable, "-B", "-S", entry, *args]
                if command["runtime"] == "python"
                else ["node", entry, *args]
            )
            print(f"CHECK {command['id']} START", flush=True)
            env = dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            tick = time.monotonic()
            result = subprocess.run(
                argv,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            log_name = command["id"] + ".log"
            (staging / log_name).write_text(
                result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8"
            )
            require(
                result.returncode == 0,
                f"Check failed: {command['id']}; exit={result.returncode}; {result.stderr[-500:]}",
            )
            checks.append(
                {
                    "id": command["id"],
                    "status": "passed",
                    "returncode": result.returncode,
                    "seconds": round(time.monotonic() - tick, 3),
                    "entry": command["entry"],
                    "args": command.get("args", []),
                    "log": log_name,
                }
            )
            print(f"CHECK {command['id']} PASS", flush=True)
        require(before == source_hashes(root, project), "Reviewed source changed during checks")
        receipt = {
            "schema_version": 1,
            "repository": project["repository"],
            "status": "passed",
            "scope": SCOPE,
            "utc": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
            "with_node": with_node,
            "seconds": round(time.monotonic() - started, 3),
            "source_sha256": before,
            "checks": checks,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "index.html").write_text(page(project, receipt, root), encoding="utf-8")
        files = {
            p.relative_to(staging).as_posix(): digest(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        (staging / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException as error:
        failure = output.with_name(output.name + "-failed-" + str(time.time_ns()))
        (staging / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error), "checks": checks}, indent=2),
            encoding="utf-8",
        )
        staging.rename(failure)
        print(f"FAILED_CHECK_EVIDENCE: {failure}", file=sys.stderr)
        raise
    verify(output, root, project, with_node)
    return {"status": "passed", "reused": False, "checks": len(checks), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-node", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Verify an existing receipt without executing checks"
    )
    parser.add_argument(
        "--render", action="store_true", help="Regenerate the tracked static inspection brief"
    )
    parser.add_argument(
        "--check-page", action="store_true", help="Reject a stale tracked inspection brief"
    )
    parser.add_argument(
        "--timeout", type=int, default=90, help="Maximum seconds per selected command"
    )
    args = parser.parse_args()
    try:
        project = load_project(ROOT)
        if args.render or args.check_page:
            target = ROOT / "review/index.html"
            rendered = page(project)
            if args.render:
                target.write_text(rendered, encoding="utf-8")
            require(target.read_text(encoding="utf-8") == rendered, "Static review page is stale")
            print("REVIEW_PAGE_VERIFIED")
            return 0
        require(not args.check or args.output is not None, "--check requires --output")
        output = args.output or Path(tempfile.gettempdir()) / ("ml-review-" + str(time.time_ns()))
        output = output.resolve()
        if args.check:
            receipt = verify(output, ROOT, project, args.with_node)
            result = {
                "status": "passed",
                "reused": True,
                "checks": len(receipt["checks"]),
                "output": str(output),
            }
        else:
            result = execute(ROOT, project, output, args.with_node, args.timeout)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(f"REVIEW_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
