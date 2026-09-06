"""Rebuild the offline public report from checksum-verified run summaries."""

from pathlib import Path

from lava.evaluation.reporting import write_report
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> int:
    """Build the report with UTC progress, heartbeat, and total elapsed time."""
    logger = RuntimeEventLogger("oracle_reader.report")
    with logger.stage("report", heartbeat_seconds=15):
        path = write_report(Path(__file__).resolve().parents[1])
    print(path)
    print("ORACLE_READER_REPORT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
