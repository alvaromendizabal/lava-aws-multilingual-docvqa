"""Execute analysis notebooks, optionally retaining checksum-verified resumable outputs."""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

from lava.notebook_execution import analysis_input_digest, execute_and_save
from lava.notebook_support import find_repo_root, git_snapshot
from lava.observability import EventLogger, ProgressReporter


def parse_args() -> argparse.Namespace:
    """Parse notebook paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        help="Save executed notebooks under artifacts/; reruns verify and reuse them",
    )
    return parser.parse_args()


def execute_notebook(
    path: Path, root: Path, timeout_seconds: int, *, kernel_name: str = "lava"
) -> nbformat.NotebookNode:
    """Run a fresh kernel; Linux IPC avoids unencrypted TCP and cleans up on failure."""
    notebook = nbformat.read(path, as_version=4)
    with TemporaryDirectory(prefix="lava-kernel-") as socket_directory:
        client = NotebookClient(
            notebook,
            timeout=timeout_seconds,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
        )
        if os.name == "posix":
            manager = client.create_kernel_manager()
            manager.transport = "ipc"
            manager.ip = str(Path(socket_directory) / "kernel")
        client.execute()
    return notebook


def main() -> int:
    """Execute or resume analysis with visible progress and optional durable local files."""
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    output_dir = (root / args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None and (root / "artifacts").resolve() not in output_dir.parents:
        raise ValueError("Saved notebook outputs must live inside repository artifacts/")
    logger = EventLogger.to_stdout(
        run_id="headless-notebook-smoke",
        component="notebook.smoke",
        jsonl_path=output_dir / "events.jsonl" if output_dir else None,
    )
    input_digest = analysis_input_digest(root) if output_dir else ""
    revision = str(git_snapshot(root)["git_commit_sha"]) if output_dir else ""
    progress = ProgressReporter(
        logger=logger,
        total=len(args.notebooks),
        event_prefix="notebooks",
    )
    started = time.perf_counter()
    for notebook_value in args.notebooks:
        path = (root / notebook_value).resolve()
        if root not in path.parents:
            message = f"Notebook path escapes repository root: {path}"
            raise ValueError(message)
        with logger.stage("notebook_execute", heartbeat_seconds=15.0, notebook=path.name):
            if output_dir is None:
                execute_notebook(path, root, args.timeout_seconds)
            else:
                saved = execute_and_save(
                    path,
                    output_dir,
                    input_digest=input_digest,
                    code_revision=revision,
                    execute=partial(execute_notebook, path, root, args.timeout_seconds),
                )
                logger.emit("notebook.output.verified", artifact=saved)
        progress.advance(notebook=path.name)
    logger.emit(
        "notebook_smoke.completed",
        notebook_count=len(args.notebooks),
        total_elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    print("HEADLESS_NOTEBOOK_SMOKE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
