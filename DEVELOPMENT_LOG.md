# Development log

## 2026-09-25 — formal benchmark protocol layer

- Added the pinned two-material formal catalog for SRBench 2025, SRSD-Feynman,
  ODEBench, and the independent legacy-classic stratum.
- Added formal seed, method-adapter, and finite-resource ledgers. The registry
  requires baseline-first execution and MySR last.
- Added deterministic clean/output-noise data processing for ground-truth and
  black-box tasks, with split checks, checksums, target visibility policy, and
  overwrite protection.
- Added `formal-validate`, `formal-seeds`, and `formal-noise` CLI commands plus
  `formal-plan`, external archive verification script, protocol tests, and documentation.
- Validation: `34 passed` with the MySR development environment; the external
  archive contains 315 tabular tasks and five ODEBench conditions covering 63
  systems. No external
  solver campaign or MySR result is claimed; raw third-party data remains
  outside this repository pending provenance and license review.

## 2026-09-26 — Execute external solver work units

- Replaced placeholder design with six real isolated solver adapters, train-only
  fits, validation selection, retained native candidates and held-out scoring.
- Added process-group time/RSS supervision, separate startup/search/scoring
  stages, resumable immutable requests and AI-Feynman Pareto timeout recovery.
- Added trajectory-separated finite-difference ODE extension, component budget
  shares, verified node-local runtime staging, and Slurm validation launchers.
- Fixed the seed validator's erroneous requirement that pilot seeds overlap
  formal seeds; the formal ledger is unchanged and new pilot seeds are disjoint.
- Documented native evaluation/complexity differences, DSR compatibility fixes,
  TF4SR applicability/pretraining limits and uncomputed symbolic/rollout metrics.
- Validation and deployed job IDs are recorded in the external deployment audit.

- Launch timing audit: first-fit PySR JIT was still inside the native fit clock.
  Added a synthetic-data warmup before the search stage, with fresh estimator/RNG
  for the actual task. Only PySR arrays are restarted; earlier records remain in
  the external superseded-run audit. The other five methods are unchanged.
