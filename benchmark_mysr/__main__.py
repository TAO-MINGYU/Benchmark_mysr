"""Command-line interface for the formal benchmark protocol."""

from __future__ import annotations

import argparse
import json
import sysconfig
from pathlib import Path

from .external_data import prepare_noisy_variants
from .formal import (
    load_formal_catalog,
    load_method_registry,
    load_resource_policy,
    load_seed_ledger,
)
from .formal_plan import build_formal_run_plan, write_formal_run_plan
from .manifests import BenchmarkDefinitionError


def _default_repository_root() -> Path:
    candidates = (
        Path.cwd(),
        Path(__file__).resolve().parents[1],
        Path(sysconfig.get_path("data")) / "share" / "benchmark_mysr",
    )
    for candidate in candidates:
        if (candidate / "manifests" / "formal" / "formal-benchmark-v1.json").is_file():
            return candidate
    return Path.cwd()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmark_mysr")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repository_root(),
        help="Benchmark_mysr repository root",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "formal-validate", help="validate the formal catalog, seed, method, and resource manifests"
    )
    subparsers.add_parser("formal-seeds", help="print the registered formal seed ledger")
    formal_plan_parser = subparsers.add_parser("formal-plan", help="write the formal baseline-first run matrix")
    formal_plan_parser.add_argument("--output", type=Path, required=True)
    noise_parser = subparsers.add_parser(
        "formal-noise", help="create clean and paired output-noise task variants"
    )
    noise_parser.add_argument("--input", type=Path, required=True, help="directory containing train/validation/test CSV files")
    noise_parser.add_argument("--spec", type=Path, required=True, help="task specification JSON")
    noise_parser.add_argument("--output", type=Path, required=True)
    noise_parser.add_argument("--noise-seed", type=int, default=0)
    noise_parser.add_argument(
        "--include-extension",
        action="store_true",
        help="also generate the predeclared heteroscedastic calibration variant",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "formal-validate":
            catalog = load_formal_catalog(args.repo_root)
            ledger = load_seed_ledger(args.repo_root)
            registry = load_method_registry(args.repo_root)
            resources = load_resource_policy(args.repo_root)
            result = {
                "benchmark_id": catalog["benchmark_id"],
                "status": "valid",
                "source_count": len(catalog["sources"]),
                "material_count": len(catalog["materials"]),
                "method_ids": [item["method_id"] for item in registry["methods"]],
                "formal_seed_count": len(ledger["formal_search_seeds"]),
                "pilot_seed_count": len(ledger["pilot_seeds"]),
                "resource_tracks": sorted(resources["tracks"]),
            }
        elif args.command == "formal-seeds":
            ledger = load_seed_ledger(args.repo_root)
            result = ledger
        elif args.command == "formal-plan":
            result = write_formal_run_plan(args.repo_root, args.output)
        elif args.command == "formal-noise":
            result = prepare_noisy_variants(
                args.input,
                args.spec,
                args.output,
                noise_seed=args.noise_seed,
                include_extension=args.include_extension,
            )
        else:
            raise BenchmarkDefinitionError(f"unsupported command: {args.command}")
    except (BenchmarkDefinitionError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
