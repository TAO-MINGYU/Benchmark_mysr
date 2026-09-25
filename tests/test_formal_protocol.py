from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_mysr.external_data import prepare_noisy_variants
from benchmark_mysr.formal import (
    formal_named_seed,
    load_formal_catalog,
    load_method_registry,
    load_resource_policy,
    load_seed_ledger,
)
from benchmark_mysr.formal_plan import build_formal_run_plan
from benchmark_mysr.manifests import BenchmarkDefinitionError
from benchmark_mysr.methods import not_applicable_result, validate_result_record


ROOT = Path(__file__).resolve().parents[1]


def test_formal_manifests_are_pinned_and_baseline_first() -> None:
    catalog = load_formal_catalog(ROOT)
    seeds = load_seed_ledger(ROOT)
    registry = load_method_registry(ROOT)
    resources = load_resource_policy(ROOT)

    assert catalog["benchmark_id"] == "mysr-formal-sr-v1"
    assert len(seeds["formal_search_seeds"]) == 10
    assert len(seeds["pilot_seeds"]) == 5
    assert [item["run_order"] for item in registry["methods"]] == list(range(1, 8))
    assert registry["methods"][-1]["method_id"] == "mysr"
    assert resources["tracks"]["constrained_resource"]["threads"] == 1
    assert resources["tracks"]["capability_ceiling"]["evaluation_limit"] > resources["tracks"]["constrained_resource"]["evaluation_limit"]


def test_formal_named_seed_is_stable_and_stream_specific() -> None:
    assert formal_named_seed(17, "task", "noise") == formal_named_seed(17, "task", "noise")
    assert formal_named_seed(17, "task", "noise") != formal_named_seed(17, "task", "search")


def test_not_applicable_result_satisfies_adapter_contract() -> None:
    record = not_applicable_result("ai_feynman_2", seed=23654, reason="black-box task")
    validate_result_record(record)
    record.pop("failure_reason")
    with pytest.raises(BenchmarkDefinitionError):
        validate_result_record(record)


def test_formal_plan_is_baseline_first_and_has_four_conditions() -> None:
    plan = build_formal_run_plan(ROOT)
    assert plan["condition_count"] == 16  # 2 materials x 4 noise x 2 resources
    assert plan["planned_run_count"] == 16 * 7 * 10
    assert plan["baseline_methods"] == ["pysr", "operon", "dsr", "ai_feynman_2", "gplearn", "tf4sr"]
    assert plan["final_method"] == "mysr"
    assert all(item["phase"] == "final_mysr_runs" for item in plan["runs"] if item["method_id"] == "mysr")


def _write_task(root: Path, *, ground_truth: bool) -> Path:
    input_dir = root / "input"
    input_dir.mkdir()
    for split, rows in {
        "train": [("tr0", 0, 1), ("tr1", 1, 3), ("tr2", 2, 5)],
        "validation": [("va0", 3, 7), ("va1", 4, 9)],
        "test": [("te0", 5, 11), ("te1", 6, 13)],
    }.items():
        (input_dir / f"{split}.csv").write_text(
            "row_id,x,target\n" + "\n".join(f"{row_id},{x},{y}" for row_id, x, y in rows) + "\n",
            encoding="utf-8",
        )
    spec = root / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "task_id": "task-test",
                "ground_truth_available": ground_truth,
                "test_target_policy": "clean_latent" if ground_truth else "observed",
            }
        ),
        encoding="utf-8",
    )
    return input_dir


def _targets(path: Path) -> list[str]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [row["target"] for row in csv.DictReader(handle)]


def test_noise_variants_are_paired_and_deterministic(tmp_path: Path) -> None:
    input_dir = _write_task(tmp_path, ground_truth=True)
    spec = tmp_path / "spec.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_noisy_variants(input_dir, spec, first, noise_seed=123)
    prepare_noisy_variants(input_dir, spec, second, noise_seed=123)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert _targets(first / "clean" / "train.csv") != _targets(first / "output_noise_10" / "train.csv")
    assert _targets(first / "clean" / "test.csv") != _targets(first / "output_noise_10" / "test.csv")
    assert (first / "clean" / "test.csv").read_text() != (first / "output_noise_10" / "test.csv").read_text()
    with pytest.raises(BenchmarkDefinitionError):
        prepare_noisy_variants(input_dir, spec, first, noise_seed=123)


def test_black_box_noise_leaves_test_observation_unchanged(tmp_path: Path) -> None:
    input_dir = _write_task(tmp_path, ground_truth=False)
    spec = tmp_path / "spec.json"
    output = tmp_path / "output"
    prepare_noisy_variants(input_dir, spec, output, noise_seed=123)
    assert (output / "clean" / "test.csv").read_bytes() == (output / "output_noise_10" / "test.csv").read_bytes()
    assert _targets(output / "clean" / "train.csv") != _targets(output / "output_noise_10" / "train.csv")


def test_extension_noise_is_explicit_and_separate(tmp_path: Path) -> None:
    input_dir = _write_task(tmp_path, ground_truth=True)
    output = tmp_path / "output"
    prepare_noisy_variants(input_dir, tmp_path / "spec.json", output, noise_seed=123, include_extension=True)
    metadata = json.loads((output / "extension_heteroscedastic" / "manifest.json").read_text())
    assert metadata["noise_model"] == "heteroscedastic_output_gaussian"
    assert metadata["noise_level"] == 0.05
