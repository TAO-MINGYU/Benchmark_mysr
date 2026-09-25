"""Formal benchmark protocol, catalog, and seed-ledger validation.

The formal catalog deliberately contains references and policies rather than
third-party raw data.  This keeps licensing and provenance review explicit and
makes a benchmark run reconstructible from pinned source revisions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifests import BenchmarkDefinitionError


FORMAL_DIRECTORY = Path("manifests") / "formal"
FORMAL_CATALOG = FORMAL_DIRECTORY / "formal-benchmark-v1.json"
SEED_LEDGER = FORMAL_DIRECTORY / "seed-ledger-v1.json"
METHOD_REGISTRY = FORMAL_DIRECTORY / "method-registry-v1.json"
RESOURCE_POLICY = FORMAL_DIRECTORY / "resource-policy-v1.json"

EXPECTED_METHODS = (
    "mysr",
    "pysr",
    "operon",
    "dsr",
    "ai_feynman_2",
    "gplearn",
    "tf4sr",
)
EXPECTED_NOISE_VARIANTS = ("clean", "output_noise_01", "output_noise_05", "output_noise_10")
_HEX40 = set("0123456789abcdef")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BenchmarkDefinitionError(f"formal manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkDefinitionError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkDefinitionError(f"formal manifest root must be an object: {path}")
    return value


def _inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BenchmarkDefinitionError(f"formal manifest path escapes repository: {path}") from error
    return resolved


def _required(mapping: dict[str, Any], names: tuple[str, ...], context: str) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise BenchmarkDefinitionError(f"{context} is missing: {', '.join(missing)}")


def _sha256_revision(revision: str, context: str) -> None:
    if len(revision) != 40 or set(revision.lower()) - _HEX40:
        raise BenchmarkDefinitionError(f"{context}.revision must be a 40-character hexadecimal commit")


def load_formal_catalog(repository_root: Path, path: Path | None = None) -> dict[str, Any]:
    """Load and validate the formal catalog and its companion ledgers."""

    root = repository_root.resolve()
    catalog_path = path or (root / FORMAL_CATALOG)
    catalog = _load_object(catalog_path.resolve())
    validate_formal_catalog(catalog)
    return catalog


def load_seed_ledger(repository_root: Path) -> dict[str, Any]:
    ledger = _load_object(_inside(repository_root.resolve(), SEED_LEDGER))
    validate_seed_ledger(ledger)
    return ledger


def load_method_registry(repository_root: Path) -> dict[str, Any]:
    registry = _load_object(_inside(repository_root.resolve(), METHOD_REGISTRY))
    validate_method_registry(registry)
    return registry


def load_resource_policy(repository_root: Path) -> dict[str, Any]:
    policy = _load_object(_inside(repository_root.resolve(), RESOURCE_POLICY))
    validate_resource_policy(policy)
    return policy


def validate_formal_catalog(catalog: dict[str, Any]) -> None:
    _required(
        catalog,
        ("schema_version", "benchmark_id", "sources", "materials", "methods", "tracks", "noise_policy", "resource_policy", "execution_order"),
        "formal catalog",
    )
    if catalog["schema_version"] != "formal-benchmark-v1":
        raise BenchmarkDefinitionError("formal catalog has unsupported schema_version")
    sources = catalog["sources"]
    if not isinstance(sources, list) or len(sources) < 4:
        raise BenchmarkDefinitionError("formal catalog requires SRBench, SRSD, ODEBench, and legacy sources")
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise BenchmarkDefinitionError("formal catalog source must be an object")
        _required(source, ("source_id", "url", "revision", "license", "license_status"), "formal source")
        source_id = source["source_id"]
        if not isinstance(source_id, str) or source_id in source_ids:
            raise BenchmarkDefinitionError(f"duplicate or invalid source_id: {source_id!r}")
        source_ids.add(source_id)
        if source_id != "legacy-classic":
            _sha256_revision(source["revision"], f"source {source_id}")
    materials = catalog["materials"]
    if not isinstance(materials, list) or {item.get("material_id") for item in materials if isinstance(item, dict)} != {"A", "B"}:
        raise BenchmarkDefinitionError("formal catalog materials must contain exactly material A and B")
    method_ids = tuple(item.get("method_id") for item in catalog["methods"] if isinstance(item, dict))
    if set(method_ids) != set(EXPECTED_METHODS) or len(method_ids) != len(EXPECTED_METHODS):
        raise BenchmarkDefinitionError(f"formal catalog methods must be {EXPECTED_METHODS}")
    if not isinstance(catalog["tracks"], dict) or set(catalog["tracks"]) != {"common_space", "native_default"}:
        raise BenchmarkDefinitionError("formal catalog must define common_space and native_default tracks")
    common_protocol = catalog.get("common_protocol")
    if not isinstance(common_protocol, dict):
        raise BenchmarkDefinitionError("formal catalog must define common_protocol")
    _required(common_protocol, ("split_names", "operator_set", "complexity_metric", "primary_metrics", "preprocessing", "invalid_prediction_policy"), "common protocol")
    if common_protocol["split_names"] != ["train", "validation", "test"]:
        raise BenchmarkDefinitionError("common protocol split names must be train, validation, test")
    if not common_protocol["operator_set"] or not isinstance(common_protocol["operator_set"], list):
        raise BenchmarkDefinitionError("common protocol operator_set must be non-empty")
    noise = catalog["noise_policy"]
    if tuple(noise.get("variants", ())) != EXPECTED_NOISE_VARIANTS:
        raise BenchmarkDefinitionError("formal catalog noise variants are not the registered clean/noise levels")
    if noise.get("sigma_definition") != "noise_level * std(y_train_clean)":
        raise BenchmarkDefinitionError("formal catalog has an unexpected output-noise sigma definition")
    if noise.get("seed_stream") != "noise":
        raise BenchmarkDefinitionError("noise must use its independent noise seed stream")
    resource = catalog["resource_policy"]
    if set(resource.get("tracks", ())) != {"constrained_resource", "capability_ceiling"}:
        raise BenchmarkDefinitionError("formal catalog must define two resource tracks")
    if resource.get("formal_seeds") != 10 or resource.get("pilot_seeds") != 5:
        raise BenchmarkDefinitionError("formal catalog seed counts must be 10 formal and 5 pilot")
    expected_order = ["catalog_and_data_processing", "baseline_adapters_and_pilot", "protocol_freeze", "baseline_runs", "mysr_code_quality_window", "final_mysr_runs"]
    if catalog["execution_order"] != expected_order:
        raise BenchmarkDefinitionError("formal execution order is not baseline-first and MySR-last")


def validate_seed_ledger(ledger: dict[str, Any]) -> None:
    _required(ledger, ("schema_version", "pilot_seeds", "formal_search_seeds", "deterministic_repeats", "streams"), "seed ledger")
    if ledger["schema_version"] != "seed-ledger-v1":
        raise BenchmarkDefinitionError("unsupported seed ledger schema_version")
    pilot = ledger["pilot_seeds"]
    formal = ledger["formal_search_seeds"]
    if not isinstance(pilot, list) or len(pilot) != 5 or len(set(pilot)) != 5:
        raise BenchmarkDefinitionError("seed ledger requires five unique pilot seeds")
    if not isinstance(formal, list) or len(formal) != 10 or len(set(formal)) != 10:
        raise BenchmarkDefinitionError("seed ledger requires ten unique formal search seeds")
    if not set(pilot).issubset(formal):
        raise BenchmarkDefinitionError("pilot seeds must be a subset of formal search seeds")
    if any(not isinstance(seed, int) or seed < 0 for seed in formal):
        raise BenchmarkDefinitionError("search seeds must be non-negative integers")
    if ledger["deterministic_repeats"] != 2:
        raise BenchmarkDefinitionError("deterministic repeat count must be two")
    _required(ledger["streams"], ("dataset", "split", "noise", "search", "restart"), "seed streams")


def validate_method_registry(registry: dict[str, Any]) -> None:
    _required(registry, ("schema_version", "result_contract", "methods", "not_applicable_policy"), "method registry")
    if registry["schema_version"] != "method-registry-v1":
        raise BenchmarkDefinitionError("unsupported method registry schema_version")
    required_result_fields = {"selected_expression", "full_frontier", "train_score", "validation_score", "test_score", "complexity", "runtime", "cpu_time", "peak_memory", "evaluations", "failure_status", "seed"}
    if set(registry["result_contract"]) != required_result_fields:
        raise BenchmarkDefinitionError("method registry result contract is incomplete")
    methods = registry["methods"]
    ids = tuple(item.get("method_id") for item in methods if isinstance(item, dict))
    if set(ids) != set(EXPECTED_METHODS) or len(ids) != len(EXPECTED_METHODS):
        raise BenchmarkDefinitionError(f"method registry must contain {EXPECTED_METHODS}")
    orders = sorted(item.get("run_order") for item in methods)
    if orders != list(range(1, 8)) or next(item["method_id"] for item in methods if item["run_order"] == 7) != "mysr":
        raise BenchmarkDefinitionError("method registry must run six baselines before MySR")


def validate_resource_policy(policy: dict[str, Any]) -> None:
    _required(policy, ("schema_version", "tracks", "calibration", "failure_states"), "resource policy")
    if policy["schema_version"] != "resource-policy-v1":
        raise BenchmarkDefinitionError("unsupported resource policy schema_version")
    tracks = policy["tracks"]
    if set(tracks) != {"constrained_resource", "capability_ceiling"}:
        raise BenchmarkDefinitionError("resource policy must define constrained_resource and capability_ceiling")
    for name, track in tracks.items():
        _required(track, ("wall_seconds", "evaluation_limit", "memory_limit_gib", "threads"), f"resource track {name}")
        if any(not isinstance(track[key], (int, float)) or track[key] <= 0 for key in ("wall_seconds", "evaluation_limit", "memory_limit_gib")):
            raise BenchmarkDefinitionError(f"resource track {name} has invalid finite limits")
        if track["threads"] != 1:
            raise BenchmarkDefinitionError("formal serial track requires one thread")


def formal_named_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable independent stream seed without Python hash randomization."""

    payload = "|".join((str(base_seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def formal_manifest_paths(repository_root: Path) -> dict[str, Path]:
    root = repository_root.resolve()
    return {
        "catalog": _inside(root, FORMAL_CATALOG),
        "seeds": _inside(root, SEED_LEDGER),
        "methods": _inside(root, METHOD_REGISTRY),
        "resources": _inside(root, RESOURCE_POLICY),
    }
