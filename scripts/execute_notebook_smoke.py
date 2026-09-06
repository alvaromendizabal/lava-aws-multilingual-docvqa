"""Execute selected notebooks headlessly without persisting their outputs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from lava.notebook_support import find_repo_root
from lava.observability import EventLogger, ProgressReporter


def parse_args() -> argparse.Namespace:
    """Parse notebook paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    """Execute notebooks in memory and report progress without saving output."""
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    logger = EventLogger.to_stdout(run_id="headless-notebook-smoke", component="notebook.smoke")
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
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=args.timeout_seconds,
                kernel_name="lava",
                resources={"metadata": {"path": str(root)}},
                allow_errors=False,
            )
            client.execute()
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
