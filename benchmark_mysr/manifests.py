"""Shared error type for formal benchmark manifest validation."""

from __future__ import annotations


class BenchmarkDefinitionError(ValueError):
    """Raised when a formal benchmark manifest violates its contract."""
