"""Build deterministic private assets for gold-evidence reader benchmarking."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pymupdf as fitz
from botocore.exceptions import ClientError
from PIL import Image

from lava.evaluation.normalization import parse_evidence_pages
from lava.evaluation.schemas import AnswerFormat, ReferenceRecord
from lava.readers.schemas import OracleExample, OraclePageAsset

_REQUIRED_COLUMNS = {
    "id",
    "file_id",
    "question",
    "answer_format",
    "answer",
    "evidence_page_number",
    "language",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checksum_sha256(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def parse_training_csv(payload: bytes) -> tuple[ReferenceRecord, ...]:
    """Parse private LAVA labels into validated references."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    columns = set(reader.fieldnames or ())
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")
    records = tuple(
        ReferenceRecord(
            question_id=row["id"],
            document_id=row["file_id"],
            question=row["question"],
            answer_format=AnswerFormat.from_raw(row["answer_format"]),
            answer=row["answer"],
            evidence_pages=parse_evidence_pages(row["evidence_page_number"]),
            language=row["language"],
        )
        for row in reader
    )
    if len(records) != len({record.question_id for record in records}):
        raise ValueError("Training question IDs must be unique")
    return records


def document_aliases(records: tuple[ReferenceRecord, ...]) -> dict[str, str]:
    """Create deterministic public-safe aliases for source document IDs."""
    return {
        document_id: f"doc-{index:02d}"
        for index, document_id in enumerate(
            sorted({record.document_id for record in records}),
            start=1,
        )
    }


def _render_page(
    page: fitz.Page,
    *,
    dpi: int,
    max_long_edge: int,
) -> tuple[bytes, int, int]:
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    long_edge = max(image.size)
    if long_edge > max_long_edge:
        ratio = max_long_edge / long_edge
        target = (
            max(1, round(image.width * ratio)),
            max(1, round(image.height * ratio)),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), image.width, image.height


def _page_layout(page: fitz.Page) -> tuple[bytes, int, int, int]:
    words = [
        {
            "x0": round(float(row[0]), 3),
            "y0": round(float(row[1]), 3),
            "x1": round(float(row[2]), 3),
            "y1": round(float(row[3]), 3),
            "text": str(row[4]),
            "block": int(row[5]),
            "line": int(row[6]),
            "word": int(row[7]),
        }
        for row in page.get_text("words", sort=True)
    ]
    raw = page.get_text("dict", sort=True)
    blocks: list[dict[str, Any]] = []
    embedded_image_count = 0
    for index, block in enumerate(raw.get("blocks", [])):
        block_type = int(block.get("type", -1))
        bbox = [round(float(value), 3) for value in block.get("bbox", (0, 0, 0, 0))]
        if block_type == 0:
            lines: list[str] = []
            for line in block.get("lines", []):
                text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
                if text:
                    lines.append(text)
            blocks.append(
                {
                    "index": index,
                    "type": "text",
                    "bbox": bbox,
                    "text": "\n".join(lines),
                }
            )
        elif block_type == 1:
            embedded_image_count += 1
            blocks.append({"index": index, "type": "image", "bbox": bbox})
    payload = {
        "page_width_points": round(float(page.rect.width), 3),
        "page_height_points": round(float(page.rect.height), 3),
        "rotation": int(page.rotation),
        "words": words,
        "blocks": blocks,
    }
    text_block_count = sum(block["type"] == "text" for block in blocks)
    return _canonical_json(payload), len(words), text_block_count, embedded_image_count


def _head_or_none(s3_client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = error.response.get("Error", {}).get("Code")
        if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _put_if_changed(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    metadata: dict[str, str],
) -> tuple[str, str | None]:
    digest = _sha256(payload)
    head = _head_or_none(s3_client, bucket=bucket, key=key)
    if head and head.get("Metadata", {}).get("sha256") == digest:
        return "reused", head.get("VersionId")
    response = s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        ChecksumSHA256=_checksum_sha256(payload),
        Metadata={**metadata, "sha256": digest},
    )
    return "uploaded", response.get("VersionId")


def _load_protocol_lock(path: Path, expected_lock_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_lock_id") != expected_lock_id:
        raise ValueError("Oracle configuration does not match the frozen evaluation lock")
    return payload


def prepare_oracle_assets(
    *,
    bucket: str,
    region: str,
    config: dict[str, Any],
    output_dir: Path,
    protocol_lock_path: Path = Path("configs/evaluation_protocol.lock.json"),
) -> dict[str, Any]:
    """Render unique gold pages and create a private, versioned manifest."""
    _load_protocol_lock(protocol_lock_path, config["protocol_lock_id"])
    s3 = boto3.client("s3", region_name=region)
    train_response = s3.get_object(Bucket=bucket, Key="raw/kaggle/train.csv")
    train_payload = train_response["Body"].read()
    records = parse_training_csv(train_payload)
    aliases = document_aliases(records)
    asset_config = config["asset_builder"]
    profile_name = asset_config["active_render_profile"]
    render = asset_config["render_profiles"][profile_name]
    config_sha = _sha256(_canonical_json(asset_config))
    page_assets: dict[tuple[str, int], OraclePageAsset] = {}
    status_counts: Counter[str] = Counter()

    pages_by_document: dict[str, set[int]] = {}
    for record in records:
        pages_by_document.setdefault(record.document_id, set()).update(record.evidence_pages)

    for document_id, page_numbers in sorted(pages_by_document.items()):
        raw_key = f"raw/kaggle/train_pdfs/train_pdfs/{document_id}.pdf"
        pdf_response = s3.get_object(Bucket=bucket, Key=raw_key)
        pdf_payload = pdf_response["Body"].read()
        pdf_sha = _sha256(pdf_payload)
        pdf_version_id = pdf_response.get("VersionId")
        pdf_etag = str(pdf_response.get("ETag", "")).strip('"') or None
        with fitz.open(stream=pdf_payload, filetype="pdf") as document:
            for page_number in sorted(page_numbers):
                if page_number > document.page_count:
                    raise ValueError(
                        f"Evidence page {page_number} exceeds {document_id} page count"
                    )
                page = document.load_page(page_number - 1)
                native_text = page.get_text("text", sort=True)
                image_payload, width, height = _render_page(
                    page,
                    dpi=int(render["dpi"]),
                    max_long_edge=int(render["max_long_edge"]),
                )
                text_payload = native_text.encode()
                layout_payload, word_count, text_blocks, image_count = _page_layout(page)
                base = (
                    f"{asset_config['s3_prefix']}/pages/{document_id}/"
                    f"page-{page_number:04d}/{profile_name}"
                )
                metadata = {
                    "source-pdf-sha256": pdf_sha,
                    "asset-config-sha256": config_sha,
                    "page-number": str(page_number),
                    "render-profile": profile_name,
                    "protocol-lock-id": config["protocol_lock_id"],
                }
                image_status, image_version = _put_if_changed(
                    s3,
                    bucket=bucket,
                    key=f"{base}/page.png",
                    payload=image_payload,
                    content_type="image/png",
                    metadata=metadata,
                )
                text_status, text_version = _put_if_changed(
                    s3,
                    bucket=bucket,
                    key=f"{base}/native_text.txt",
                    payload=text_payload,
                    content_type="text/plain; charset=utf-8",
                    metadata=metadata,
                )
                layout_status, layout_version = _put_if_changed(
                    s3,
                    bucket=bucket,
                    key=f"{base}/layout.json",
                    payload=layout_payload,
                    content_type="application/json",
                    metadata=metadata,
                )
                status_counts.update((image_status, text_status, layout_status))
                page_assets[(document_id, page_number)] = OraclePageAsset(
                    asset_version=asset_config["version"],
                    document_id=document_id,
                    document_alias=aliases[document_id],
                    page_number=page_number,
                    source_pdf_s3_uri=f"s3://{bucket}/{raw_key}",
                    source_pdf_sha256=pdf_sha,
                    source_pdf_version_id=pdf_version_id,
                    source_pdf_etag=pdf_etag,
                    image_s3_uri=f"s3://{bucket}/{base}/page.png",
                    image_sha256=_sha256(image_payload),
                    image_version_id=image_version,
                    text_s3_uri=f"s3://{bucket}/{base}/native_text.txt",
                    text_sha256=_sha256(text_payload),
                    text_version_id=text_version,
                    layout_s3_uri=f"s3://{bucket}/{base}/layout.json",
                    layout_sha256=_sha256(layout_payload),
                    layout_version_id=layout_version,
                    width_pixels=width,
                    height_pixels=height,
                    dpi=int(render["dpi"]),
                    native_text_characters=len(native_text),
                    word_count=word_count,
                    text_block_count=text_blocks,
                    embedded_image_count=image_count,
                )

    examples = tuple(
        OracleExample(
            protocol_lock_id=config["protocol_lock_id"],
            question_id=record.question_id,
            document_id=record.document_id,
            document_alias=aliases[record.document_id],
            question=record.question,
            answer_format=record.answer_format,
            answer=record.answer,
            evidence_pages=record.evidence_pages,
            language=record.language,
            pages=tuple(page_assets[(record.document_id, page)] for page in record.evidence_pages),
        )
        for record in sorted(records, key=lambda item: item.question_id)
    )
    private_payload = (
        "\n".join(
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for example in examples
        )
        + "\n"
    ).encode()
    manifest_sha = _sha256(private_payload)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_key = f"{asset_config['s3_prefix']}/manifests/{run_id}/oracle_examples.jsonl"
    latest_key = f"{asset_config['s3_prefix']}/manifests/latest/oracle_examples.jsonl"
    manifest_versions: dict[str, str | None] = {}
    for key in (manifest_key, latest_key):
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=private_payload,
            ContentType="application/x-ndjson",
            ChecksumSHA256=_checksum_sha256(private_payload),
            Metadata={
                "sha256": manifest_sha,
                "protocol-lock-id": config["protocol_lock_id"],
                "asset-config-sha256": config_sha,
            },
        )
        manifest_versions[key] = response.get("VersionId")

    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = Path("artifacts/oracle_reader/oracle_examples.jsonl")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(private_payload)
    summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_lock_id": config["protocol_lock_id"],
        "question_count": len(examples),
        "document_count": len(aliases),
        "unique_evidence_page_count": len(page_assets),
        "language_counts": dict(sorted(Counter(item.language for item in examples).items())),
        "answer_format_counts": dict(
            sorted(Counter(item.answer_format.value for item in examples).items())
        ),
        "evidence_cardinality_counts": dict(
            sorted(Counter(str(len(item.evidence_pages)) for item in examples).items())
        ),
        "render": {
            "profile": profile_name,
            "dpi": int(render["dpi"]),
            "max_long_edge": int(render["max_long_edge"]),
            "format": asset_config["image_format"],
            "asset_config_sha256": config_sha,
        },
        "private_manifest_sha256": manifest_sha,
        "private_manifest_s3_uri": f"s3://{bucket}/{latest_key}",
        "private_manifest_version_id": manifest_versions[latest_key],
        "object_status_counts": dict(sorted(status_counts.items())),
        "privacy_boundary": (
            "Questions, answers, source IDs, page images, text, and layout stay in private S3."
        ),
    }
    summary_path = output_dir / "oracle_assets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path = output_dir / "ORACLE_ASSETS.md"
    report_path.write_text(
        "# Oracle-Evidence Assets\n\n"
        f"- Protocol lock: `{summary['protocol_lock_id']}`\n"
        f"- Questions: **{summary['question_count']}**\n"
        f"- Documents: **{summary['document_count']}**\n"
        f"- Unique evidence pages: **{summary['unique_evidence_page_count']}**\n"
        f"- Languages: `{json.dumps(summary['language_counts'], sort_keys=True)}`\n"
        f"- Answer formats: `{json.dumps(summary['answer_format_counts'], sort_keys=True)}`\n"
        f"- Evidence cardinalities: `{json.dumps(summary['evidence_cardinality_counts'], sort_keys=True)}`\n"
        f"- Render: `{json.dumps(summary['render'], sort_keys=True)}`\n\n"
        "Private labels and document contents are not committed to GitHub.\n",
        encoding="utf-8",
    )
    for path, key in (
        (
            summary_path,
            f"{config['benchmark']['public_reports_prefix']}/latest/{summary_path.name}",
        ),
        (report_path, f"{config['benchmark']['public_reports_prefix']}/latest/{report_path.name}"),
    ):
        s3.upload_file(str(path), bucket, key)
    return summary


def load_oracle_examples(payload: bytes) -> tuple[OracleExample, ...]:
    """Load a private newline-delimited oracle-example manifest."""
    records = [OracleExample.model_validate_json(line) for line in payload.splitlines() if line]
    return tuple(records)
