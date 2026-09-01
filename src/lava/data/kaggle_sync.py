
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

FIELDS = ("name", "size", "creationDate")
TOKEN_PREFIX = "Next Page Token = "


@contextmanager
def kaggle_auth(parameter: str, region: str) -> Iterator[None]:
    auth_dir = Path.home() / ".kaggle"
    auth_file = auth_dir / "access_token"
    previous = auth_file.read_bytes() if auth_file.exists() else None
    value = boto3.client("ssm", region_name=region).get_parameter(
        Name=parameter,
        WithDecryption=True,
    )["Parameter"]["Value"]
    auth_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    auth_file.write_text(str(value).strip(), encoding="utf-8")
    auth_file.chmod(0o600)
    try:
        yield
    finally:
        if previous is None:
            auth_file.unlink(missing_ok=True)
        else:
            auth_file.write_bytes(previous)
            auth_file.chmod(0o600)


def parse_page(text: str) -> tuple[list[dict[str, str]], str | None]:
    token = None
    csv_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(TOKEN_PREFIX):
            token = line.removeprefix(TOKEN_PREFIX).strip() or None
        else:
            csv_lines.append(raw)
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise RuntimeError(f"Unexpected Kaggle columns: {reader.fieldnames}")
    rows = []
    for row in reader:
        if not row["name"] or not str(row["size"]).isdigit():
            raise RuntimeError(f"Invalid Kaggle row: {row}")
        rows.append({field: str(row[field]).strip() for field in FIELDS})
    return rows, token


def list_files(kaggle: str, competition: str) -> list[dict[str, str]]:
    found = {}
    page_token = None
    seen_tokens = set()
    page = 0
    while True:
        page += 1
        command = [
            kaggle,
            "competitions",
            "files",
            competition,
            "--page-size=200",
            "-v",
            "-q",
        ]
        if page_token:
            command.extend(["--page-token", page_token])
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        rows, next_token = parse_page(result.stdout)
        print(f"PAGE {page}: {len(rows)} files", flush=True)
        for row in rows:
            prior = found.get(row["name"])
            if prior is not None and prior != row:
                raise RuntimeError(f"Conflicting metadata for {row['name']}")
            found[row["name"]] = row
        if not next_token:
            break
        if next_token in seen_tokens:
            raise RuntimeError("Repeated Kaggle page token.")
        seen_tokens.add(next_token)
        page_token = next_token
    return [found[name] for name in sorted(found)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_inventory(
    rows: list[dict[str, str]],
    competition: str,
) -> dict[str, object]:
    inventory_path = Path("reports/kaggle_files.csv")
    write_csv(inventory_path, rows, list(FIELDS))
    total_bytes = sum(int(row["size"]) for row in rows)
    summary = {
        "competition": competition,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 4),
        "inventory_sha256": sha256(inventory_path),
        "top_level_counts": dict(
            sorted(Counter(row["name"].split("/", 1)[0] for row in rows).items())
        ),
        "extensions": dict(
            sorted(Counter(Path(row["name"]).suffix.lower() for row in rows).items())
        ),
        "largest_files": [
            {
                "name": row["name"],
                "size_mib": round(int(row["size"]) / 1024**2, 2),
            }
            for row in sorted(
                rows,
                key=lambda item: int(item["size"]),
                reverse=True,
            )[:10]
        ],
    }
    summary_path = Path("reports/kaggle_inventory_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def existing_digest(s3, bucket: str, key: str, expected_size: int) -> str | None:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    metadata = head.get("Metadata", {})
    digest = str(metadata.get("sha256", ""))
    if (
        int(head["ContentLength"]) == expected_size
        and metadata.get("source-size") == str(expected_size)
        and len(digest) == 64
    ):
        return digest
    return None


def download_payload(
    kaggle: str,
    competition: str,
    name: str,
    expected_size: int,
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    last_error = None
    for attempt in range(1, 4):
        temporary = tempfile.TemporaryDirectory(prefix="lava-", dir="/tmp")
        directory = Path(temporary.name)
        try:
            subprocess.run(
                [
                    kaggle,
                    "competitions",
                    "download",
                    competition,
                    "-f",
                    name,
                    "-p",
                    str(directory),
                    "-o",
                    "-q",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            files = [path for path in directory.rglob("*") if path.is_file()]
            candidates = [
                path for path in files if path.stat().st_size == expected_size
            ]
            if len(candidates) == 1:
                return candidates[0], temporary

            requested_basename = Path(name).name
            for archive_path in files:
                if not zipfile.is_zipfile(archive_path):
                    continue
                with zipfile.ZipFile(archive_path) as archive:
                    members = [
                        member
                        for member in archive.infolist()
                        if not member.is_dir()
                        and (
                            member.filename == name
                            or Path(member.filename).name == requested_basename
                        )
                    ]
                    if len(members) != 1:
                        continue
                    extracted = directory / "extracted" / requested_basename
                    extracted.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.open(members[0]) as source,
                        extracted.open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
                    if extracted.stat().st_size == expected_size:
                        return extracted, temporary

            all_files = [
                (str(path.relative_to(directory)), path.stat().st_size)
                for path in files
            ]
            raise RuntimeError(
                f"Expected one {expected_size}-byte payload for {name}; "
                f"found {all_files}"
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.SubprocessError,
            zipfile.BadZipFile,
        ) as error:
            last_error = error
            temporary.cleanup()
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Download failed for {name}") from last_error


def sync_files(
    rows: list[dict[str, str]],
    *,
    kaggle: str,
    competition: str,
    bucket: str,
    region: str,
    limit: int,
) -> dict[str, object]:
    selected = rows if limit == 0 else rows[:limit]
    complete = len(selected) == len(rows)
    s3 = boto3.client("s3", region_name=region)
    manifest = []
    for number, row in enumerate(selected, start=1):
        name = row["name"]
        size = int(row["size"])
        key = f"raw/kaggle/{name}"
        digest = existing_digest(s3, bucket, key, size)
        status = "reused"
        if digest is None:
            print(f"[{number}/{len(selected)}] DOWNLOAD {name}", flush=True)
            payload, temporary = download_payload(
                kaggle,
                competition,
                name,
                size,
            )
            try:
                digest = sha256(payload)
                s3.upload_file(
                    str(payload),
                    bucket,
                    key,
                    ExtraArgs={
                        "Metadata": {
                            "sha256": digest,
                            "source-size": str(size),
                            "source-creation-date": row["creationDate"],
                            "competition": competition,
                        }
                    },
                )
            finally:
                temporary.cleanup()
            verified = existing_digest(s3, bucket, key, size)
            if verified != digest:
                raise RuntimeError(f"S3 verification failed for {key}")
            status = "uploaded"
        else:
            print(f"[{number}/{len(selected)}] REUSE {name}", flush=True)
        head = s3.head_object(Bucket=bucket, Key=key)
        manifest.append(
            {
                **row,
                "sha256": digest,
                "s3_key": key,
                "version_id": str(head.get("VersionId", "")),
                "status": status,
                "verified_at_utc": datetime.now(UTC).isoformat(),
            }
        )

    suffix = "" if complete else "_smoke"
    manifest_path = Path(f"reports/raw_data_manifest{suffix}.csv")
    fields = [
        "name",
        "size",
        "creationDate",
        "sha256",
        "s3_key",
        "version_id",
        "status",
        "verified_at_utc",
    ]
    write_csv(manifest_path, manifest, fields)
    summary = {
        "complete": complete,
        "expected_file_count": len(rows),
        "verified_file_count": len(manifest),
        "verified_bytes": sum(int(row["size"]) for row in manifest),
        "verified_gib": round(
            sum(int(row["size"]) for row in manifest) / 1024**3,
            4,
        ),
        "manifest_sha256": sha256(manifest_path),
        "uploaded_count": sum(row["status"] == "uploaded" for row in manifest),
        "reused_count": sum(row["status"] == "reused" for row in manifest),
    }
    summary_path = Path(f"reports/raw_data_manifest_summary{suffix}.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"manifests/ingestion/{run_id}"
    s3.upload_file(str(manifest_path), bucket, f"{prefix}/{manifest_path.name}")
    s3.upload_file(str(summary_path), bucket, f"{prefix}/{summary_path.name}")
    if complete:
        s3.upload_file(
            str(manifest_path),
            bucket,
            "manifests/latest/raw_data_manifest.csv",
        )
        s3.upload_file(
            str(summary_path),
            bucket,
            "manifests/latest/raw_data_manifest_summary.json",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="lava-challenge-2026")
    parser.add_argument("--parameter", default="/lava/kaggle/api-token")
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle executable was not found.")

    with kaggle_auth(args.parameter, args.region):
        rows = list_files(kaggle, args.competition)
        inventory_summary = write_inventory(rows, args.competition)
        s3 = boto3.client("s3", region_name=args.region)
        s3.upload_file(
            "reports/kaggle_files.csv",
            args.bucket,
            "manifests/source/kaggle_files.csv",
        )
        s3.upload_file(
            "reports/kaggle_inventory_summary.json",
            args.bucket,
            "manifests/source/kaggle_inventory_summary.json",
        )
        print(json.dumps(inventory_summary, indent=2, sort_keys=True))
        if args.sync:
            sync_summary = sync_files(
                rows,
                kaggle=kaggle,
                competition=args.competition,
                bucket=args.bucket,
                region=args.region,
                limit=args.limit,
            )
            print(json.dumps(sync_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

