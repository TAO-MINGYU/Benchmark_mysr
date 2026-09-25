"""Adapter metadata and result contract for formal comparison methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .formal import load_method_registry
from .manifests import BenchmarkDefinitionError


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    adapter: str
    supports: tuple[str, ...]
    native_environment: str
    run_order: int
    not_applicable_reason: str | None = None

    def supports_stratum(self, stratum: str) -> bool:
        return stratum in self.supports


class SolverAdapter(Protocol):
    """Runtime adapter boundary used by the future execution runner."""

    method_id: str

    def run(self, task: dict[str, Any], *, seed: int, resource_track: str) -> dict[str, Any]:
        ...


RESULT_FIELDS = (
    "selected_expression",
    "full_frontier",
    "train_score",
    "validation_score",
    "test_score",
    "complexity",
    "runtime",
    "cpu_time",
    "peak_memory",
    "evaluations",
    "failure_status",
    "seed",
)


def method_specs(repository_root) -> tuple[MethodSpec, ...]:
    registry = load_method_registry(repository_root)
    return tuple(
        MethodSpec(
            method_id=item["method_id"],
            adapter=item["adapter"],
            supports=tuple(item["supports"]),
            native_environment=item["native_environment"],
            run_order=item["run_order"],
            not_applicable_reason=item.get("not_applicable_reason"),
        )
        for item in sorted(registry["methods"], key=lambda value: value["run_order"])
    )


def not_applicable_result(method_id: str, *, seed: int, reason: str) -> dict[str, Any]:
    """Return a complete result record for a method/task mismatch."""

    return {
        "method_id": method_id,
        "selected_expression": None,
        "full_frontier": [],
        "train_score": None,
        "validation_score": None,
        "test_score": None,
        "complexity": None,
        "runtime": 0.0,
        "cpu_time": 0.0,
        "peak_memory": None,
        "evaluations": 0,
        "failure_status": "not_applicable",
        "failure_reason": reason,
        "seed": seed,
    }


def validate_result_record(record: dict[str, Any]) -> None:
    """Validate one adapter result before it enters the metrics ledger."""

    missing = [field for field in RESULT_FIELDS if field not in record]
    if missing:
        raise BenchmarkDefinitionError(f"adapter result is missing: {', '.join(missing)}")
    if not isinstance(record["seed"], int) or record["seed"] < 0:
        raise BenchmarkDefinitionError("adapter result seed must be a non-negative integer")
    status = record["failure_status"]
    if status not in {"success", "timeout", "memory_limit", "crash", "invalid_output", "not_applicable"}:
        raise BenchmarkDefinitionError(f"unknown adapter failure_status: {status}")
    if not isinstance(record["evaluations"], int) or record["evaluations"] < 0:
        raise BenchmarkDefinitionError("adapter result evaluations must be a non-negative integer")
    if status == "not_applicable" and not record.get("failure_reason"):
        raise BenchmarkDefinitionError("not_applicable adapter result requires failure_reason")
