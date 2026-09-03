"""Fail-closed error types. Every gate violation raises one of these; the CLI
maps them to a clear stderr message and a non-zero exit. No silent fallbacks.
"""

from __future__ import annotations


class AmlWorkbenchError(Exception):
    """Base class for all fail-closed workbench errors."""


class DownloadError(AmlWorkbenchError):
    """Every download channel for a dataset failed (or drift was detected)."""


class DataQualityError(AmlWorkbenchError):
    """Checksum, byte-size, schema, or count assertion violation."""


class SmokeGateError(AmlWorkbenchError):
    """Smoke metrics below the gate or runtime over the limit."""
