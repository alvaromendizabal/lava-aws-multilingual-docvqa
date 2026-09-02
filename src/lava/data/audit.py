"""Resumable data audit for the LAVA multilingual document-VQA corpus."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pymupdf
from botocore.exceptions import BotoCoreError, ClientError

NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "<na>"}
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
VIETNAMESE_RE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệ"
    r"ìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?%?$")
INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON atomically."""
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_cell(value: Any) -> str:
    """Normalize a CSV cell for profiling."""
    return "" if value is None else str(value).strip()


def is_nullish(value: Any) -> bool:
    """Return whether a value is empty or a common null marker."""
    return clean_cell(value).lower() in NULL_TOKENS


def classify_language(text: str, fallback: str = "unknown") -> str:
    """Classify Japanese or Vietnamese using script-level evidence."""
    if JAPANESE_RE.search(text):
        return "ja"
    if VIETNAMESE_RE.search(text):
        return "vi"
    return fallback


def classify_answer_type(value: Any) -> str:
    """Classify a labeled answer without exposing its content."""
    text = clean_cell(value)
    if is_nullish(text):
        return "missing"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return "list"
    if isinstance(parsed, int | float) and not isinstance(parsed, bool):
        return "number"
    if NUMBER_RE.fullmatch(text.replace(" ", "")):
        return "number"
    if any(separator in text for separator in ("|", ";", "\n")):
        return "list_like"
    word_count = len(text.split())
    return "short_text" if word_count <= 12 else "long_text"


def parse_page_references(value: Any) -> list[int]:
    """Parse page references from JSON, Python literals, or free text."""
    text = clean_cell(value)
    if is_nullish(text):
        return []
    parsed: Any = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (ValueError, SyntaxError):
            continue
    candidates: list[Any]
    if isinstance(parsed, list | tuple | set):
        candidates = list(parsed)
    elif isinstance(parsed, int | float) and not isinstance(parsed, bool):
        candidates = [parsed]
    else:
        candidates = INTEGER_RE.findall(text)
    pages: list[int] = []
    for candidate in candidates:
        try:
            page = int(candidate)
        except (TypeError, ValueError):
            continue
        if page >= 0 and page not in pages:
            pages.append(page)
    return pages


def value_kind(value: Any) -> str:
    """Return a lightweight, content-free value type."""
    text = clean_cell(value)
    if is_nullish(text):
        return "missing"
    if NUMBER_RE.fullmatch(text.replace(" ", "")):
        return "numeric"
    if text.startswith(("[", "{")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "string"
        return type(parsed).__name__
    return "string"


def fallback_language_from_key(key: str) -> str:
    """Infer corpus language from the PDF identifier prefix."""
    stem = Path(key).stem.lower()
    if stem.startswith("j_"):
        return "ja"
    if stem.startswith("v_"):
        return "vi"
    return "unknown"


def split_from_key(key: str) -> str:
    """Infer train or test split from an S3 object key."""
    lowered = key.lower()
    if "/train_pdfs/" in lowered or lowered.endswith("/train.csv"):
        return "train"
    if "/test_pdfs/" in lowered or lowered.endswith("/test.csv"):
        return "test"
    if lowered.endswith("/sample_submission.csv"):
        return "sample_submission"
    return "unknown"


def quantile(values: list[float], probability: float) -> float | None:
    """Return a linearly interpolated quantile."""
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] * (1 - fraction) + clean[upper] * fraction


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    """Return stable descriptive statistics."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "p25": quantile(clean, 0.25),
        "median": quantile(clean, 0.50),
        "p75": quantile(clean, 0.75),
        "max": max(clean),
        "mean": statistics.fmean(clean),
    }


def iter_s3_objects(s3: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """List all non-placeholder S3 objects under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    rows: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/.keep"):
                continue
            rows.append(
                {
                    "s3_key": key,
                    "size_bytes": int(item["Size"]),
                    "etag": str(item.get("ETag", "")).strip('"'),
                    "last_modified_utc": item["LastModified"].astimezone(UTC).isoformat(),
                }
            )
    return sorted(rows, key=lambda row: row["s3_key"])


def decode_csv_bytes(payload: bytes) -> str:
    """Decode a competition CSV with explicit fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def infer_semantic_columns(columns: list[str]) -> dict[str, list[str]]:
    """Infer candidate document, question, answer, and evidence fields."""
    lowered = {column: column.lower() for column in columns}
    return {
        "document": [
            column
            for column, name in lowered.items()
            if any(token in name for token in ("pdf", "document", "doc", "file"))
        ],
        "question": [
            column
            for column, name in lowered.items()
            if any(token in name for token in ("question", "query"))
        ],
        "answer": [
            column
            for column, name in lowered.items()
            if "answer" in name
            and not any(token in name for token in ("format", "type", "page", "evidence"))
        ],
        "evidence": [
            column
            for column, name in lowered.items()
            if any(token in name for token in ("evidence", "page", "support"))
        ],
        "identifier": [
            column for column, name in lowered.items() if name == "id" or name.endswith("_id")
        ],
    }


def profile_csv(key: str, payload: bytes, known_pdf_stems: set[str]) -> dict[str, Any]:
    """Profile a CSV while retaining no raw question or answer values."""
    text = decode_csv_bytes(payload)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    columns = list(reader.fieldnames or [])
    profiles: dict[str, Any] = {}
    for column in columns:
        values = [clean_cell(row.get(column)) for row in rows]
        nonmissing = [value for value in values if not is_nullish(value)]
        profiles[column] = {
            "nonmissing_count": len(nonmissing),
            "missing_count": len(values) - len(nonmissing),
            "unique_count": len(set(nonmissing)),
            "maximum_character_length": max((len(value) for value in nonmissing), default=0),
            "value_kind_counts": dict(
                sorted(Counter(value_kind(value) for value in values).items())
            ),
        }

    inferred = infer_semantic_columns(columns)
    document_match_counts: dict[str, int] = {}
    for column in columns:
        count = 0
        for row in rows:
            value = clean_cell(row.get(column))
            if Path(value).stem.lower() in known_pdf_stems:
                count += 1
        if count:
            document_match_counts[column] = count
    if document_match_counts:
        ranked = sorted(
            document_match_counts,
            key=lambda column: document_match_counts[column],
            reverse=True,
        )
        inferred["document"] = list(dict.fromkeys(ranked + inferred["document"]))

    question_column = inferred["question"][0] if inferred["question"] else None
    answer_column = inferred["answer"][0] if inferred["answer"] else None
    evidence_column = inferred["evidence"][0] if inferred["evidence"] else None
    document_column = inferred["document"][0] if inferred["document"] else None

    languages: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    evidence_cardinalities: Counter[str] = Counter()
    questions: list[str] = []
    referenced_documents: list[str] = []
    for row in rows:
        fallback = "unknown"
        if document_column:
            document_value = clean_cell(row.get(document_column))
            referenced_documents.append(Path(document_value).stem.lower())
            fallback = fallback_language_from_key(document_value)
        if question_column:
            question = clean_cell(row.get(question_column))
            questions.append(question)
            languages[classify_language(question, fallback)] += 1
        elif fallback != "unknown":
            languages[fallback] += 1
        if answer_column:
            answer_types[classify_answer_type(row.get(answer_column))] += 1
        if evidence_column:
            evidence_cardinalities[str(len(parse_page_references(row.get(evidence_column))))] += 1

    nonempty_questions = [question for question in questions if question]
    referenced_set = {value for value in referenced_documents if value}
    return {
        "s3_key": key,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "column_profiles": profiles,
        "inferred_columns": inferred,
        "selected_columns": {
            "document": document_column,
            "question": question_column,
            "answer": answer_column,
            "evidence": evidence_column,
        },
        "language_counts": dict(sorted(languages.items())),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "evidence_page_cardinality_counts": dict(sorted(evidence_cardinalities.items())),
        "duplicate_question_row_count": len(nonempty_questions) - len(set(nonempty_questions)),
        "referenced_document_count": len(referenced_set),
        "missing_referenced_documents": sorted(referenced_set - known_pdf_stems),
    }


def audit_pdf(
    s3: Any,
    bucket: str,
    item: dict[str, Any],
    temporary_directory: Path,
) -> dict[str, Any]:
    """Download and audit one PDF, then remove the temporary file."""
    key = str(item["s3_key"])
    start = time.perf_counter()
    local_name = hashlib.sha256(key.encode()).hexdigest() + ".pdf"
    local_path = temporary_directory / local_name
    base: dict[str, Any] = {
        **item,
        "split": split_from_key(key),
        "language": fallback_language_from_key(key),
        "source_sha256": "",
        "computed_sha256": "",
        "sha256_matches_metadata": None,
        "status": "error",
        "error_type": "",
        "error_message": "",
    }
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        object_metadata = {
            str(name).lower(): str(value) for name, value in head.get("Metadata", {}).items()
        }
        source_sha256 = (
            object_metadata.get("sha256")
            or object_metadata.get("source-sha256")
            or object_metadata.get("source_sha256")
            or ""
        )
        base["source_sha256"] = source_sha256
        s3.download_file(bucket, key, str(local_path))
        actual_size = local_path.stat().st_size
        if actual_size != int(item["size_bytes"]):
            msg = f"Downloaded size {actual_size} differs from S3 size {item['size_bytes']}"
            raise ValueError(msg)
        computed_sha256 = sha256_file(local_path)
        base["computed_sha256"] = computed_sha256
        base["sha256_matches_metadata"] = not source_sha256 or computed_sha256 == source_sha256
        if source_sha256 and computed_sha256 != source_sha256:
            raise ValueError("Downloaded SHA-256 differs from S3 object metadata")

        with pymupdf.open(local_path) as document:
            page_count = int(document.page_count)
            is_encrypted = bool(document.is_encrypted)
            needs_password = bool(document.needs_pass)
            if needs_password:
                raise PermissionError("PDF requires a password")
            text_chars: list[int] = []
            word_counts: list[int] = []
            block_counts: list[int] = []
            image_counts: list[int] = []
            widths: list[float] = []
            heights: list[float] = []
            for page in document:
                text = page.get_text("text", sort=True)
                text_chars.append(len(re.sub(r"\s+", "", text)))
                word_counts.append(len(page.get_text("words")))
                block_counts.append(len(page.get_text("blocks")))
                image_counts.append(len(page.get_images(full=True)))
                widths.append(float(page.rect.width))
                heights.append(float(page.rect.height))
            pages_with_native_text = sum(count >= 20 for count in text_chars)
            pages_with_images = sum(count > 0 for count in image_counts)
            text_page_ratio = pages_with_native_text / page_count if page_count else 0.0
            if text_page_ratio >= 0.90:
                document_class = "native_text_dominant"
            elif text_page_ratio <= 0.10:
                document_class = "image_or_scan_dominant"
            else:
                document_class = "mixed_text_and_image"
            metadata = document.metadata or {}
            base.update(
                {
                    "status": "ok",
                    "pdf_format": clean_cell(metadata.get("format")),
                    "is_encrypted": is_encrypted,
                    "needs_password": needs_password,
                    "page_count": page_count,
                    "pages_with_native_text": pages_with_native_text,
                    "pages_without_native_text": page_count - pages_with_native_text,
                    "native_text_page_ratio": round(text_page_ratio, 6),
                    "pages_with_images": pages_with_images,
                    "image_page_ratio": (
                        round(pages_with_images / page_count, 6) if page_count else 0.0
                    ),
                    "total_text_characters": sum(text_chars),
                    "total_word_tokens": sum(word_counts),
                    "total_text_blocks": sum(block_counts),
                    "total_embedded_images": sum(image_counts),
                    "median_text_characters_per_page": quantile(
                        [float(value) for value in text_chars], 0.5
                    ),
                    "median_words_per_page": quantile([float(value) for value in word_counts], 0.5),
                    "median_page_width_points": quantile(widths, 0.5),
                    "median_page_height_points": quantile(heights, 0.5),
                    "document_content_class": document_class,
                    "toc_entry_count": len(document.get_toc(simple=True)),
                    "metadata_title_present": bool(clean_cell(metadata.get("title"))),
                }
            )
    except (BotoCoreError, ClientError, OSError, RuntimeError, ValueError) as error:
        base["error_type"] = type(error).__name__
        base["error_message"] = str(error)[:500]
    finally:
        local_path.unlink(missing_ok=True)
        base["processing_seconds"] = round(time.perf_counter() - start, 4)
        base["audited_at_utc"] = datetime.now(UTC).isoformat()
    return base


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load a JSON-lines checkpoint keyed by S3 object key."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["s3_key"])] = row
    return rows


def save_checkpoint(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the PDF checkpoint."""
    serialized = "".join(
        json.dumps(rows[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(rows)
    )
    atomic_write_text(path, serialized)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a UTF-8 LF-terminated CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_summary(
    pdf_rows: list[dict[str, Any]],
    csv_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate public, content-free audit statistics."""
    ok_rows = [row for row in pdf_rows if row.get("status") == "ok"]
    errors = [row for row in pdf_rows if row.get("status") != "ok"]
    by_split = Counter(str(row.get("split", "unknown")) for row in ok_rows)
    by_language = Counter(str(row.get("language", "unknown")) for row in ok_rows)
    by_content_class = Counter(str(row.get("document_content_class", "unknown")) for row in ok_rows)
    sha_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        digest = clean_cell(row.get("computed_sha256"))
        if digest:
            sha_groups[digest].append(row)
    duplicate_groups = [group for group in sha_groups.values() if len(group) > 1]
    cross_split_duplicate_groups = [
        group for group in duplicate_groups if len({str(row.get("split")) for row in group}) > 1
    ]
    total_pages = sum(int(row.get("page_count", 0)) for row in ok_rows)
    pages_with_text = sum(int(row.get("pages_with_native_text", 0)) for row in ok_rows)
    pages_with_images = sum(int(row.get("pages_with_images", 0)) for row in ok_rows)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pdf_count": len(pdf_rows),
        "pdf_success_count": len(ok_rows),
        "pdf_error_count": len(errors),
        "pdf_counts_by_split": dict(sorted(by_split.items())),
        "pdf_counts_by_language": dict(sorted(by_language.items())),
        "document_content_class_counts": dict(sorted(by_content_class.items())),
        "total_pages": total_pages,
        "pages_with_native_text": pages_with_text,
        "native_text_page_ratio": (
            round(pages_with_text / total_pages, 6) if total_pages else None
        ),
        "pages_with_embedded_images": pages_with_images,
        "embedded_image_page_ratio": (
            round(pages_with_images / total_pages, 6) if total_pages else None
        ),
        "pdf_size_bytes": summarize_numeric([float(row["size_bytes"]) for row in ok_rows]),
        "page_count": summarize_numeric([float(row["page_count"]) for row in ok_rows]),
        "native_text_page_ratio_by_document": summarize_numeric(
            [float(row["native_text_page_ratio"]) for row in ok_rows]
        ),
        "processing_seconds": summarize_numeric(
            [float(row["processing_seconds"]) for row in pdf_rows]
        ),
        "exact_duplicate_pdf_group_count": len(duplicate_groups),
        "cross_split_exact_duplicate_pdf_group_count": len(cross_split_duplicate_groups),
        "exact_duplicate_pdf_groups": [
            [str(row["s3_key"]) for row in group] for group in duplicate_groups
        ],
        "cross_split_exact_duplicate_pdf_groups": [
            [str(row["s3_key"]) for row in group] for group in cross_split_duplicate_groups
        ],
        "pdf_errors": [
            {
                "s3_key": row.get("s3_key"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in errors
        ],
        "csv_profiles": csv_profiles,
        "deferred_audits": [
            "near-duplicate page detection using perceptual or embedding similarity",
            "layout and table/figure classification",
            "OCR engine comparison",
            "page-image quality and orientation analysis",
        ],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    """Render the public data-audit report."""
    page_stats = summary["page_count"]
    size_stats = summary["pdf_size_bytes"]
    lines = [
        "# LAVA Data Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Corpus integrity",
        "",
        f"- PDFs audited: **{summary['pdf_count']}**",
        f"- PDFs opened successfully: **{summary['pdf_success_count']}**",
        f"- PDF errors: **{summary['pdf_error_count']}**",
        f"- Total pages: **{summary['total_pages']}**",
        f"- Exact duplicate PDF groups: **{summary['exact_duplicate_pdf_group_count']}**",
        (
            "- Cross-split exact duplicate groups: "
            f"**{summary['cross_split_exact_duplicate_pdf_group_count']}**"
        ),
        "",
        "## Corpus composition",
        "",
        f"- Split counts: `{json.dumps(summary['pdf_counts_by_split'], sort_keys=True)}`",
        f"- Language counts: `{json.dumps(summary['pdf_counts_by_language'], sort_keys=True)}`",
        (
            "- Content classes: "
            f"`{json.dumps(summary['document_content_class_counts'], sort_keys=True)}`"
        ),
        f"- Native-text page ratio: **{summary['native_text_page_ratio']}**",
        f"- Embedded-image page ratio: **{summary['embedded_image_page_ratio']}**",
        "",
        "## Document scale",
        "",
        (
            "- Pages per PDF: "
            f"min={page_stats['min']}, median={page_stats['median']}, "
            f"p75={page_stats['p75']}, max={page_stats['max']}"
        ),
        (
            "- PDF bytes: "
            f"min={size_stats['min']}, median={size_stats['median']}, "
            f"p75={size_stats['p75']}, max={size_stats['max']}"
        ),
        "",
        "## CSV schemas",
        "",
    ]
    for profile in summary["csv_profiles"]:
        selected = profile["selected_columns"]
        lines.extend(
            [
                f"### `{profile['s3_key']}`",
                "",
                f"- Rows: **{profile['row_count']}**",
                f"- Columns: **{profile['column_count']}**",
                (f"- Selected semantic columns: `{json.dumps(selected, sort_keys=True)}`"),
                (f"- Language counts: `{json.dumps(profile['language_counts'], sort_keys=True)}`"),
                (f"- Answer types: `{json.dumps(profile['answer_type_counts'], sort_keys=True)}`"),
                (
                    "- Evidence cardinalities: "
                    f"`{json.dumps(profile['evidence_page_cardinality_counts'], sort_keys=True)}`"
                ),
                f"- Duplicate question rows: **{profile['duplicate_question_row_count']}**",
                (
                    "- Missing referenced documents: "
                    f"**{len(profile['missing_referenced_documents'])}**"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "This phase audits source integrity, document scale, native-text availability, "
                "and label/schema structure. It does not use hidden test labels, does not tune a "
                "model, and does not claim that native text is sufficient for answering. "
                "Near-duplicate page detection, layout classification, OCR comparison, and "
                "multimodal modeling remain separate experiments so their contribution can be "
                "measured."
            ),
            "",
            "## Next research gate",
            "",
            (
                "Freeze a document-isolated validation protocol and implement the official "
                "answer-plus-evidence evaluator before training or selecting a retrieval or "
                "vision-language model."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def upload_outputs(s3: Any, bucket: str, paths: list[Path]) -> str:
    """Upload audit outputs to versioned and latest S3 prefixes."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for path in paths:
        for prefix in (f"reports/data-audit/{run_id}", "reports/data-audit/latest"):
            s3.upload_file(str(path), bucket, f"{prefix}/{path.name}")
    return run_id


def git_sha() -> str:
    """Return the current Git commit SHA when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_audit(args: argparse.Namespace) -> None:
    """Run the resumable data audit."""
    s3 = boto3.client("s3", region_name=args.region)
    objects = iter_s3_objects(s3, args.bucket, args.raw_prefix)
    pdf_items = [item for item in objects if str(item["s3_key"]).lower().endswith(".pdf")]
    csv_items = [item for item in objects if str(item["s3_key"]).lower().endswith(".csv")]
    if not pdf_items or not csv_items:
        msg = f"Expected PDFs and CSVs under s3://{args.bucket}/{args.raw_prefix}"
        raise RuntimeError(msg)

    known_pdf_stems = {Path(str(item["s3_key"])).stem.lower() for item in pdf_items}
    csv_profiles: list[dict[str, Any]] = []
    for item in csv_items:
        response = s3.get_object(Bucket=args.bucket, Key=item["s3_key"])
        csv_profiles.append(
            profile_csv(
                str(item["s3_key"]),
                response["Body"].read(),
                known_pdf_stems,
            )
        )

    artifact_dir = Path(args.artifact_dir)
    public_dir = Path(args.public_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / f"pdf_inventory_{args.mode}.jsonl"
    remote_checkpoint = f"temp/data-audit/pdf_inventory_{args.mode}.jsonl"
    if not checkpoint.exists():
        try:
            s3.download_file(args.bucket, remote_checkpoint, str(checkpoint))
            print(
                f"RESTORED CHECKPOINT s3://{args.bucket}/{remote_checkpoint}",
                flush=True,
            )
        except ClientError as error:
            error_code = str(error.response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                raise

    selected_items = pdf_items
    if args.mode == "smoke":
        train_items = [item for item in pdf_items if split_from_key(str(item["s3_key"])) == "train"]
        test_items = [item for item in pdf_items if split_from_key(str(item["s3_key"])) == "test"][
            :3
        ]
        selected_items = train_items + test_items

    completed = load_checkpoint(checkpoint)
    temporary_directory = Path(tempfile.mkdtemp(prefix="lava-data-audit-", dir="/tmp"))
    try:
        for index, item in enumerate(selected_items, start=1):
            key = str(item["s3_key"])
            prior = completed.get(key)
            reusable = (
                prior
                and prior.get("status") == "ok"
                and prior.get("etag") == item.get("etag")
                and int(prior.get("size_bytes", -1)) == int(item["size_bytes"])
            )
            if reusable:
                print(f"[{index}/{len(selected_items)}] REUSE {key}", flush=True)
                continue
            print(f"[{index}/{len(selected_items)}] AUDIT {key}", flush=True)
            completed[key] = audit_pdf(
                s3,
                args.bucket,
                item,
                temporary_directory,
            )
            save_checkpoint(checkpoint, completed)
            if index % args.checkpoint_every == 0:
                s3.upload_file(str(checkpoint), args.bucket, remote_checkpoint)
        save_checkpoint(checkpoint, completed)
        s3.upload_file(str(checkpoint), args.bucket, remote_checkpoint)
    finally:
        try:
            temporary_directory.rmdir()
        except OSError:
            pass

    selected_keys = {str(item["s3_key"]) for item in selected_items}
    pdf_rows = [completed[key] for key in sorted(selected_keys) if key in completed]
    detailed_csv = artifact_dir / f"pdf_inventory_{args.mode}.csv"
    csv_profile_path = artifact_dir / "csv_profiles.json"
    write_csv(detailed_csv, pdf_rows)
    write_json(csv_profile_path, csv_profiles)

    summary = aggregate_summary(pdf_rows, csv_profiles)
    summary.update(
        {
            "audit_mode": args.mode,
            "bucket": re.sub(r"(?<!\d)\d{12}(?!\d)", "<account-id>", args.bucket),
            "raw_prefix": args.raw_prefix,
            "git_commit_sha": git_sha(),
            "audit_source_sha256": sha256_file(Path(__file__)),
            "expected_pdf_count_for_mode": len(selected_items),
            "audited_pdf_count_for_mode": len(pdf_rows),
        }
    )
    summary_path = public_dir / f"data_audit_summary_{args.mode}.json"
    report_path = public_dir / f"DATA_AUDIT_{args.mode.upper()}.md"
    write_json(summary_path, summary)
    atomic_write_text(report_path, markdown_report(summary))
    run_id = upload_outputs(
        s3,
        args.bucket,
        [checkpoint, detailed_csv, csv_profile_path, summary_path, report_path],
    )
    print(flush=True)
    print("DATA AUDIT COMPLETE", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"S3_RUN_ID={run_id}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--raw-prefix", default="raw/kaggle/")
    parser.add_argument("--artifact-dir", default="artifacts/data_audit")
    parser.add_argument("--public-dir", default="reports/data_audit")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    return parser.parse_args()


def main() -> None:
    """Run the command-line interface."""
    run_audit(parse_args())


if __name__ == "__main__":
    main()
