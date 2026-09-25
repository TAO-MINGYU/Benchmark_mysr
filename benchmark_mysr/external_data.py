"""Prepare paired clean/noisy task directories for external benchmark sources."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .formal import formal_named_seed
from .manifests import BenchmarkDefinitionError


_LEVELS = {"clean": 0.0, "output_noise_01": 0.01, "output_noise_05": 0.05, "output_noise_10": 0.10}
_EXTENSION_VARIANT = "extension_heteroscedastic"
_REQUIRED_SPEC = ("task_id", "ground_truth_available", "test_target_policy")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise BenchmarkDefinitionError(f"invalid task specification: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkDefinitionError("task specification must be an object")
    return value


def _read_split(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise BenchmarkDefinitionError(f"split has no header: {path}")
            rows = list(reader)
    except FileNotFoundError as error:
        raise BenchmarkDefinitionError(f"split does not exist: {path}") from error
    if not rows:
        raise BenchmarkDefinitionError(f"split is empty: {path}")
    if "target" not in reader.fieldnames:
        raise BenchmarkDefinitionError(f"split must contain target column: {path}")
    if "row_id" not in reader.fieldnames:
        raise BenchmarkDefinitionError(f"split must contain row_id column: {path}")
    feature_columns = [name for name in reader.fieldnames if name not in {"row_id", "target", "target_clean", "target_observed"}]
    if not feature_columns:
        raise BenchmarkDefinitionError(f"split must contain at least one feature column: {path}")
    for row_number, row in enumerate(rows, start=2):
        for name in (*feature_columns, "target"):
            try:
                value = float(row[name])
            except (TypeError, ValueError) as error:
                raise BenchmarkDefinitionError(f"non-numeric {name} at {path}:{row_number}") from error
            if not math.isfinite(value):
                raise BenchmarkDefinitionError(f"non-finite {name} at {path}:{row_number}")
    return feature_columns, rows


def _write_split(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_values(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([float(row["target"]) for row in rows], dtype=float)


def prepare_noisy_variants(
    input_dir: Path,
    spec_path: Path,
    output_dir: Path,
    *,
    noise_seed: int = 0,
    include_extension: bool = False,
) -> dict[str, Any]:
    """Generate clean and output-noise variants without overwriting output.

    ``input_dir`` contains ``train.csv``, ``validation.csv`` and ``test.csv``.
    The source files are assumed to contain the clean latent target only when
    ``ground_truth_available`` is true.  For black-box tasks, test observations
    remain unchanged in every noise variant and exact recovery is disabled.
    """

    input_dir = input_dir.resolve()
    spec = _read_json(spec_path.resolve())
    missing = [name for name in _REQUIRED_SPEC if name not in spec]
    if missing:
        raise BenchmarkDefinitionError(f"task specification is missing: {', '.join(missing)}")
    if not isinstance(spec["task_id"], str) or not spec["task_id"]:
        raise BenchmarkDefinitionError("task_id must be a non-empty string")
    if not isinstance(spec["ground_truth_available"], bool):
        raise BenchmarkDefinitionError("ground_truth_available must be boolean")
    if spec["test_target_policy"] not in {"clean_latent", "observed"}:
        raise BenchmarkDefinitionError("test_target_policy must be clean_latent or observed")
    if output_dir.exists():
        raise BenchmarkDefinitionError(f"refusing to overwrite existing output: {output_dir}")

    splits: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for split in ("train", "validation", "test"):
        splits[split] = _read_split(input_dir / f"{split}.csv")
    feature_columns = splits["train"][0]
    if any(columns != feature_columns for columns, _ in splits.values()):
        raise BenchmarkDefinitionError("feature columns differ across splits")
    row_ids: set[str] = set()
    for split, (_, rows) in splits.items():
        for row in rows:
            row_id = row["row_id"]
            if row_id in row_ids:
                raise BenchmarkDefinitionError(f"row_id occurs in more than one split: {row_id}")
            row_ids.add(row_id)

    train_clean = _target_values(splits["train"][1])
    scale = float(np.std(train_clean))
    if not math.isfinite(scale):
        raise BenchmarkDefinitionError("training target standard deviation is non-finite")
    checksums: dict[str, str] = {}
    variant_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True)
    variants = list(_LEVELS.items())
    if include_extension:
        variants.append((_EXTENSION_VARIANT, 0.05))
    max_abs_train = max(float(np.max(np.abs(train_clean))), 1.0)
    for variant, level in variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir()
        variant_checksums: dict[str, str] = {}
        for split, (columns, source_rows) in splits.items():
            apply_noise = level > 0 and (spec["ground_truth_available"] or split != "test")
            rng = np.random.default_rng(formal_named_seed(noise_seed, spec["task_id"], variant, split))
            if variant == _EXTENSION_VARIANT:
                row_scale = 0.5 + np.abs(_target_values(source_rows)) / max_abs_train
                sigma_values = level * row_scale * scale
                noise = rng.normal(0.0, sigma_values) if apply_noise else np.zeros(len(source_rows))
                noise_model = "heteroscedastic_output_gaussian"
            else:
                noise = rng.normal(0.0, level * scale, size=len(source_rows)) if apply_noise else np.zeros(len(source_rows))
                noise_model = "homoscedastic_output_gaussian"
            fields = ["row_id", *feature_columns, "target"]
            rows: list[dict[str, Any]] = []
            for index, source_row in enumerate(source_rows):
                clean_target = float(source_row["target"])
                observed_target = clean_target + float(noise[index])
                row: dict[str, Any] = {"row_id": source_row["row_id"]}
                row.update({name: source_row[name] for name in feature_columns})
                row["target"] = f"{observed_target:.17g}"
                if spec["ground_truth_available"]:
                    fields.extend(["target_clean", "target_observed"])
                    row["target_clean"] = f"{clean_target:.17g}"
                    row["target_observed"] = f"{observed_target:.17g}"
                rows.append(row)
            # DictWriter requires unique fields; the extensions are added once.
            fields = list(dict.fromkeys(fields))
            variant_checksums[split] = _write_split(variant_dir / f"{split}.csv", fields, rows)
        metadata = {
            "task_id": spec["task_id"],
            "variant": variant,
            "noise_level": level,
            "noise_model": noise_model,
            "sigma": level * scale,
            "noise_seed": formal_named_seed(noise_seed, spec["task_id"], variant),
            "ground_truth_available": spec["ground_truth_available"],
            "test_target_policy": "clean_latent_and_observed" if spec["ground_truth_available"] else "observed_unchanged",
            "primary_robustness_target": "target_clean" if spec["ground_truth_available"] else None,
            "checksums": variant_checksums,
        }
        (variant_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksums[variant] = hashlib.sha256((variant_dir / "manifest.json").read_bytes()).hexdigest()
        variant_records.append(metadata)
    manifest = {
        "schema_version": "formal-noise-material-v1",
        "task_id": spec["task_id"],
        "input_spec": spec,
        "feature_columns": feature_columns,
        "row_counts": {split: len(rows) for split, (_, rows) in splits.items()},
        "clean_train_std": scale,
        "variants": variant_records,
        "directory_manifest_sha256": checksums,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "generated",
        "task_id": spec["task_id"],
        "output": str(output_dir),
        "variants": [variant for variant, _ in variants],
        "clean_train_std": scale,
    }
