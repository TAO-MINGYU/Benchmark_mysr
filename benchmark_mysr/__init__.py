"""Utilities for validating and running the formal MySR benchmark protocol."""

from .external_data import prepare_noisy_variants
from .formal import (
    formal_named_seed,
    load_formal_catalog,
    load_method_registry,
    load_resource_policy,
    load_seed_ledger,
)
from .formal_plan import build_formal_run_plan, write_formal_run_plan
from .manifests import BenchmarkDefinitionError
from .methods import MethodSpec, method_specs, not_applicable_result, validate_result_record

__all__ = [
    "BenchmarkDefinitionError",
    "formal_named_seed",
    "build_formal_run_plan",
    "load_formal_catalog",
    "load_method_registry",
    "load_resource_policy",
    "load_seed_ledger",
    "MethodSpec",
    "method_specs",
    "not_applicable_result",
    "prepare_noisy_variants",
    "write_formal_run_plan",
    "validate_result_record",
]

__version__ = "0.1.0.dev0"
