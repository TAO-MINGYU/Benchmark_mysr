"""Verify a materialized formal-corpus archive without running solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY_VARIANTS = {"clean", "output_noise_01", "output_noise_05", "output_noise_10"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive: Path) -> dict[str, object]:
    manifest_path = archive / "materialized-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if len(tasks) != manifest.get("task_count"):
        raise ValueError("materialized task_count does not match tasks")
    missing: list[str] = []
    for task in tasks:
        output = archive / task["output"]
        variants = {path.name for path in output.iterdir() if path.is_dir()}
        if not PRIMARY_VARIANTS.issubset(variants):
            missing.append(task["task_id"])
        for variant in PRIMARY_VARIANTS:
            variant_manifest = output / variant / "manifest.json"
            if not variant_manifest.is_file():
                missing.append(f"{task['task_id']}:{variant}:manifest")
            for split in ("train", "validation", "test"):
                if not (output / variant / f"{split}.csv").is_file():
                    missing.append(f"{task['task_id']}:{variant}:{split}")
    if missing:
        raise ValueError(f"archive is incomplete: {missing[:10]}")
    ode_manifest = archive / "odebench-manifest.json"
    ode = json.loads(ode_manifest.read_text(encoding="utf-8"))
    for condition in ode.get("conditions", []):
        path = archive / condition["file"]
        if _sha256(path) != condition["sha256"]:
            raise ValueError(f"ODEBench hash mismatch: {path}")
    return {
        "status": "valid",
        "task_count": len(tasks),
        "primary_variants": sorted(PRIMARY_VARIANTS),
        "ode_condition_count": len(ode.get("conditions", [])),
        "ode_system_counts": sorted({condition["system_count"] for condition in ode.get("conditions", [])}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
