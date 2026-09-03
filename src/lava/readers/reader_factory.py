"""Fail-closed construction of oracle-reader implementations."""

from __future__ import annotations

from typing import Protocol

from lava.readers.qwen35 import Qwen35Reader
from lava.readers.schemas import (
    OracleExample,
    ReaderFamily,
    ReaderPrediction,
    ReaderTelemetry,
    ResolvedModel,
)


class OracleReader(Protocol):
    """Structural contract shared by oracle-reader implementations."""

    def predict(
        self,
        example: OracleExample,
    ) -> tuple[ReaderPrediction, ReaderTelemetry]:
        """Generate one prediction and its telemetry."""


def build_reader(
    model_spec: ResolvedModel,
    *,
    region: str,
) -> OracleReader:
    """Construct only explicitly supported reader families."""
    if model_spec.reader_family is ReaderFamily.QWEN3_5:
        return Qwen35Reader(
            model_spec,
            region=region,
        )

    raise ValueError(f"Reader family is not implemented yet: {model_spec.reader_family.value!r}")
