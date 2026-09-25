# Formal Symbolic Regression Benchmark v1

This repository now contains the protocol layer for the formal MySR benchmark.
It pins source revisions and policies; third-party raw datasets remain external
until their licenses and checksums pass the ingest review.

## Materials

Material A is the SRBench Unified Core at the pinned SRBench 2025 revision. It
has separate ground-truth, black-box, and legacy-classic strata. Nguyen,
Keijzer, Pagie-1, Vladislavleva, and Korns are downloaded from the pinned
[Equation Recovery Benchmark](https://huggingface.co/datasets/EquationDiscovery/Equation_Recovery_Benchmark)
task table and reported as the independent legacy stratum. Material B combines
SRSD-Feynman (including dummy-variable tasks) with ODEBench for
scientific-discovery and dynamic-system robustness. The exact revisions and source URLs are in
`manifests/formal/formal-benchmark-v1.json`.

## Paired noise generation

`formal-noise` takes a task directory containing `train.csv`,
`validation.csv`, `test.csv`, and a task specification. It creates:

```text
clean/
output_noise_01/
output_noise_05/
output_noise_10/
```

For ground-truth tasks, each noisy split retains `target_clean` and
`target_observed`; the primary robustness score uses the clean latent test
target. For black-box tasks, noise is applied to train and validation only and
the observed test target is unchanged, so no exact-recovery claim is made. The
noise scale is `level * std(y_train_clean)`, and the noise stream is derived
from an explicit task/variant seed. Existing output directories are never
overwritten.

On a predeclared calibration subset only, `--include-extension` adds one
`extension_heteroscedastic` variant. Its row-wise Gaussian scale is
`0.05 * (0.5 + abs(y) / max(abs(y_train))) * std(y_train)`. It is a separate
stress condition and is not silently included in the four primary tracks.

Example:

```bash
python -m benchmark_mysr formal-noise \
  --input /path/to/task \
  --spec /path/to/task/spec.json \
  --output /home/taomingyu/taomingyu_5/MySR_Benchmark/<run-id>/task-001/noise \
  --noise-seed 123
```

The external archive `20260925-formal-corpus-v1` contains 315 standardized
tasks: 24 selected SRBench 2025 PMLB tasks, 120 published SRSD-Feynman tasks,
120 upstream-style dummy-variable tasks, and 51 legacy classic tasks (12
Nguyen, 15 Keijzer, 1 Pagie, 8 Vladislavleva, and 15 Korns). It also contains
five ODEBench trajectory conditions for all 63 systems. The archive manifests
record source revisions, generation parameters, and SHA-256 hashes. Raw data
remain external because they are third-party files and the processed corpus is
large.

## Methods and resource tracks

The method registry fixes PySR, Operon, DSR, AI-Feynman 2.0, gplearn, TF4SR, and MySR.
Every adapter must emit the same result fields and record `not_applicable`
with a reason when a method's assumptions do not fit a task. The common-space
and native-default tracks are reported separately.

Formal runs use ten fixed search seeds. Five pilot seeds are used only for
adapter and budget calibration. Two deterministic repeats are reserved for
smoke and representative tasks. The seed ledger keeps dataset, split, noise,
search, and restart streams separate.

The two resource tracks are finite and frozen before formal runs:

- `constrained_resource`: initial candidate 20,000 evaluations, 120 seconds,
  8 GiB, one thread;
- `capability_ceiling`: initial candidate 500,000 evaluations, 900 seconds,
  32 GiB, one thread.

The pilot may choose final values from the declared candidate grid
`{20k, 100k, 500k} × {120s, 300s, 900s}` using a recorded calibration rule.
It may not add method-specific exceptions after seeing formal outcomes.
Startup and compilation time are reported separately from search time.

## Execution order

The intended order is catalog/data processing, adapter pilot, protocol freeze,
all five baseline campaigns, a MySR/MySRCore code-quality window, and only then
the final MySR campaign. Baseline outcomes cannot be used to remove difficult
tasks. Historical `original-development-v0.1`, `capability-difference-v0.1`, and
`Benchmark_PySR_Shortcoming` materials are not active inputs and are not silently
promoted into this release.

Validate the protocol with:

```bash
python -m benchmark_mysr formal-validate
python -m benchmark_mysr formal-seeds
python -m benchmark_mysr formal-plan --output /tmp/formal-run-plan.json
python -m pytest -q
python scripts/verify_external_archive.py \
  --archive /home/taomingyu/taomingyu_5/MySR_Benchmark/20260925-formal-corpus-v1
```

No formal solver campaign is claimed by this repository change. A run release
must additionally include adapter commits, environment locks, raw per-seed
metrics, complete frontiers, resource logs, failure records, and provenance
hashes under the external result root.
