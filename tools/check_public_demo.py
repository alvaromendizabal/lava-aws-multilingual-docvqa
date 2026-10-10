"""Check the reviewed static demo boundary without executing its JavaScript.

This focused packaging check complements the behavioral browser-demo tests;
it is not a security sandbox or a validation of the historical research model.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

REQUIRED = {"index.html", "styles.css", "engine.js", "app.js"}
SUFFIXES = {".html", ".css", ".js", ".svg"}
SECRET = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
PRIVATE = re.compile(r"s3://|arn:aws(?:-[a-z]+)?:|/(?:home|workspace|mnt)/", re.IGNORECASE)
NETWORK = re.compile(
    r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|importScripts)\s*\("
    r"|\bsendBeacon\s*\(|\bimport\s*\("
)
IMPORT = re.compile(r"\b(?:import|export)\s+(?:[^;\n]*?\s+from\s+)?[\"']([^\"']+)[\"']")
CSS_URL = re.compile(r"url\(\s*[\"']?([^\s)\"']+)[\"']?\s*\)", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_asset(root: Path, source: Path, reference: str) -> None:
    target = urlsplit(reference)
    require(not target.scheme and not target.netloc, "Remote runtime asset: " + reference)
    if not target.path and target.fragment:
        return
    decoded = unquote(target.path)
    require(bool(decoded) and not decoded.startswith(("/", "\\")), "Invalid asset path")
    path = (source.parent / decoded).resolve()
    require(path.is_relative_to(root.resolve()), "Asset escapes demo: " + reference)
    require(path.is_file(), "Missing runtime asset: " + reference)


class RuntimeAssets(HTMLParser):
    def __init__(self, root: Path, source: Path):
        super().__init__(convert_charrefs=True)
        self.root, self.source = root, source

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        require(tag not in {"iframe", "object", "embed", "base"}, "External document container")
        if tag == "meta":
            require(attributes.get("http-equiv", "").lower() != "refresh", "Automatic redirect")
        # A canonical URL is identity metadata, not a downloaded runtime asset.
        if tag == "link" and attributes.get("rel", "").lower() == "canonical":
            reference = urlsplit(attributes.get("href") or "")
            require(reference.scheme == "https" and bool(reference.netloc), "Invalid canonical URL")
            return
        if tag in {"script", "img", "source", "audio", "video", "link"}:
            for key in ("src", "href", "poster"):
                if attributes.get(key):
                    local_asset(self.root, self.source, attributes[key])
            require("srcset" not in attributes, "Unreviewed responsive runtime asset")


def verify(root: Path) -> dict:
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing or symlinked demo directory")
    for name in REQUIRED:
        require((root / name).is_file(), "Missing demo entry point: " + name)
    files = []
    text_by_name = {}
    total = 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlink in public demo")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        require(path.suffix in SUFFIXES, "Unreviewed public demo file: " + relative)
        raw = path.read_bytes()
        total += len(raw)
        require(len(raw) <= 1_000_000 and total <= 2_000_000, "Oversized static demo")
        text = raw.decode("utf-8")
        require(not SECRET.search(text), "Credential-like material in " + relative)
        require(not PRIVATE.search(text), "Private operational locator in " + relative)
        require(not NETWORK.search(text), "Runtime network or dynamic import in " + relative)
        if path.suffix == ".html":
            RuntimeAssets(root, path).feed(text)
        if path.suffix == ".js":
            for reference in IMPORT.findall(text):
                local_asset(root, path, reference)
        if path.suffix == ".svg":
            for element in ET.fromstring(text).iter():
                require(element.tag.rsplit("}", 1)[-1] != "script", "Active SVG content")
                for key, value in element.attrib.items():
                    if key.endswith("href"):
                        local_asset(root, path, value)
        if path.suffix in {".css", ".svg", ".html"}:
            require("@import" not in text.lower(), "Unreviewed CSS import")
            for reference in CSS_URL.findall(text):
                local_asset(root, path, reference)
        text_by_name[relative] = text
        files.append(
            {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    require(len(files) <= 32, "Unexpected static demo file count")
    require("synthetic" in text_by_name["index.html"].lower(), "Missing visible synthetic scope")
    require(
        any(
            "SYNTHETIC_ONLY" in text for name, text in text_by_name.items() if name.endswith(".js")
        ),
        "Missing machine-readable synthetic evidence marker",
    )
    return {"status": "PASS", "scope": "reviewed_static_demo_only", "files": files, "bytes": total}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1] / "public-demo"), indent=2))
