"""Verify one canonical, executed notebook sequence in notebooks/."""

from pathlib import Path

from lava.notebook_execution import NOTEBOOK_STEMS, validate_public_notebook
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    logger = RuntimeEventLogger("notebooks.validation")
    with logger.stage("notebooks.validate", heartbeat_seconds=15):
        observed = {path.stem for path in (root / "notebooks").glob("*.ipynb")}
        if observed != set(NOTEBOOK_STEMS):
            raise ValueError(f"Unexpected canonical notebook set: {sorted(observed)}")
        if list((root / "notebooks").glob("*.py")) or list((root / "reports").rglob("*.ipynb")):
            raise ValueError("Duplicate notebook sources or publication copies found")
        for index, stem in enumerate(NOTEBOOK_STEMS, 1):
            validate_public_notebook(root, root / "notebooks" / f"{stem}.ipynb")
            logger.emit(
                "notebook.verified", notebook=stem, completed=index, total=len(NOTEBOOK_STEMS)
            )
    print("PUBLIC_NOTEBOOKS_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
