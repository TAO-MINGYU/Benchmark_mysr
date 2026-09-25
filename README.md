# MySR Formal Symbolic Regression Benchmark

This repository is the single versioned protocol and adapter specification for the
MySR/MySRCore formal symbolic-regression benchmark. The previous development and
PySR-shortcoming suites are no longer active benchmark inputs.

The external corpus is materialized at
`/home/taomingyu/taomingyu_5/MySR_Benchmark/20260925-formal-corpus-v1` and is
identified by the manifests and SHA-256 records in that archive. The GitHub
repository contains the reproducible code, source revisions, schemas, environment
specifications, run plans, and usage documentation. Large third-party data files
remain in the versioned external archive rather than being copied into Git history.

## Corpus

- Material A: 12 SRBench/PMLB black-box tasks, 12 first-principles source tasks,
  and 51 legacy classic equation tasks.
- Material B: 120 published SRSD-Feynman tasks, 120 deterministic dummy-variable
  variants, and ODEBench trajectory data for 63 systems.
- Primary tabular variants: `clean`, `output_noise_01`, `output_noise_05`, and
  `output_noise_10`.

See [`FORMAL_BENCHMARK.md`](FORMAL_BENCHMARK.md) and
[`BENCHMARK_USAGE_PLAN.md`](BENCHMARK_USAGE_PLAN.md) for the complete protocol.

## Methods

The fixed horizontal comparison roster is PySR, Operon, DSR, AI-Feynman 2.0,
gplearn, and TF4SR. TF4SR is the selected Transformer baseline because its
official repository includes pretrained weights and a direct SRSD evaluation
script; it is reported with its supported scientific-equation strata. MySR is
run only after all baseline campaigns and the MySR code-quality window.

The registry records both a matched common-space track and a native-default track.
Each adapter must preserve the result contract, seed ledger, resource limits, and
failure states. A method that is not applicable to a task is recorded as such; the
task set is never changed to make a method look better.

## Validation

From this repository root:

```bash
python -m pip install -e '.[test]'
python -m benchmark_mysr formal-validate
python -m benchmark_mysr formal-seeds
python -m benchmark_mysr formal-plan --output /tmp/mysr-formal-run-plan.json
python -m pytest -q
python scripts/verify_external_archive.py \
  --archive /home/taomingyu/taomingyu_5/MySR_Benchmark/20260925-formal-corpus-v1
```

No solver result is considered formal until the protocol, task membership, seeds,
resource budgets, adapters, and environment locks are frozen.

## License

Repository-authored code and documentation are licensed under Apache License 2.0.
Third-party data and task definitions retain their original licenses and notices.
