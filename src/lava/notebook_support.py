"""Shared notebook bootstrapping and public-metadata sanitization."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from lava.observability.events import sanitize_value


def find_repo_root(start: Path | None = None) -> Path:
    """Find the nearest repository root containing ``pyproject.toml`` and ``.git``."""
    candidates: list[Path] = []
    explicit = os.environ.get("LAVA_REPO_ROOT")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    current = (start or Path.cwd()).expanduser().resolve()
    candidates.extend([current, *current.parents])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    message = "Unable to locate the LAVA repository root."
    raise FileNotFoundError(message)


def load_project_environment(root: Path) -> dict[str, str]:
    """Load the ignored project ``.env`` file and validate required variables."""
    env_path = root / ".env"
    if not env_path.is_file():
        message = f"Missing project environment file: {env_path}"
        raise FileNotFoundError(message)
    load_dotenv(env_path, override=False)
    required = ("AWS_REGION", "S3_BUCKET")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        message = f"Missing required environment variables: {', '.join(missing)}"
        raise RuntimeError(message)
    return {key: os.environ[key] for key in required}


def git_snapshot(root: Path) -> dict[str, object]:
    """Return branch, commit, and worktree state for a notebook run."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "git_commit_sha": run("rev-parse", "HEAD"),
        "working_tree_clean": not status,
    }


def public_metadata(value: Mapping[str, object]) -> dict[str, object]:
    """Drop private paths and redact infrastructure identifiers for public display."""
    blocked = {
        "bucket",
        "manifest_s3_uri",
        "output_s3_prefix",
        "private_manifest_s3_uri",
        "private_manifest_version_id",
    }
    selected = {key: item for key, item in value.items() if key not in blocked}
    sanitized = sanitize_value(selected)
    if not isinstance(sanitized, dict):
        message = "Public metadata must remain a mapping."
        raise TypeError(message)
    return sanitized
