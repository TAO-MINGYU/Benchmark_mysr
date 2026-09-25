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
