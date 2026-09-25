"""Build a deterministic formal-run matrix without executing solvers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .formal import load_formal_catalog, load_method_registry, load_resource_policy, load_seed_ledger


def build_formal_run_plan(repository_root: Path) -> dict[str, Any]:
    """Return the pre-registration matrix for the four primary conditions.

    Task membership is intentionally left as an input to the material ingest
    stage.  This matrix cannot silently create or remove source tasks.
    """

    catalog = load_formal_catalog(repository_root)
    ledger = load_seed_ledger(repository_root)
    registry = load_method_registry(repository_root)
    resources = load_resource_policy(repository_root)
    materials = [item["material_id"] for item in catalog["materials"]]
    variants = list(catalog["noise_policy"]["variants"])
    tracks = list(resources["tracks"])
    methods = [item["method_id"] for item in sorted(registry["methods"], key=lambda item: item["run_order"])]
    runs = []
    for material in materials:
        for variant in variants:
            for resource_track in tracks:
                for method in methods:
                    phase = "final_mysr_runs" if method == "mysr" else "baseline_runs"
                    for seed in ledger["formal_search_seeds"]:
                        runs.append(
                            {
                                "material_id": material,
                                "noise_variant": variant,
                                "resource_track": resource_track,
                                "method_id": method,
                                "seed": seed,
                                "phase": phase,
                                "status": "planned",
                            }
                        )
    return {
        "schema_version": "formal-run-plan-v1",
        "benchmark_id": catalog["benchmark_id"],
        "task_membership": "resolved only after source ingest and provenance audit",
        "baseline_methods": [method for method in methods if method != "mysr"],
        "final_method": "mysr",
        "condition_count": len(materials) * len(variants) * len(tracks),
        "planned_run_count": len(runs),
        "runs": runs,
    }


def write_formal_run_plan(repository_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run plan: {output}")
    plan = build_formal_run_plan(repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "planned",
        "output": str(output),
        "condition_count": plan["condition_count"],
        "planned_run_count": plan["planned_run_count"],
        "baseline_methods": plan["baseline_methods"],
        "final_method": plan["final_method"],
    }
