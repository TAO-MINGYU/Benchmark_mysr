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

## 2026-09-26 — Full-node logical CPU campaign scheduling

- Added a two-node pool launcher with 512 distinct single-CPU workunit slots per
  node, round-robin admission across method/track queues, bounded startup rate,
  memory-aware admission and durable workunit/CPU/job event records.
- Original immutable solver releases, requests, budgets and completed result
  identities remain unchanged; interrupted attempts are archived before restart.
- Documented SMT concurrency as an explicit resource epoch requiring matched
  interpretation in later MySR comparisons, not an algorithmic change.
- Live full-node startup exposed supervisor NFS import stalls. Added local staging
  for the unchanged supervisor dependency closure and immutable adapter releases;
  preserved the first scheduling attempt and its interrupted output for audit.
- A live I/O probe measured 10.86 seconds to create four shared log files. Moved
  pool bookkeeping to retained local files with asynchronous status/event mirrors
  and completed-console-log publication, keeping solver result storage unchanged.

## 2026-09-28 — b3 after b1 failure audit

- Parameterized the full-node pool's CPU allocation so continuation can use all
  currently available logical CPUs without waiting for unrelated node work.
- b3 reuses b1 successes and protocol-defined outcomes, retries only missing
  workunits plus startup/scoring timeout records, and leaves explicit search/
  evaluation/memory outcomes and algorithmic failures for audit.

## 2026-10-14 — Targeted external result repair

- Preserved the canonical relocation-only changes in the isolated repair worktree.
- Fixed native expression export (DSR unsimplified tokens and gplearn iterative
  traversal), public SRSD unit alignment including permuted dummy columns, and
  invalid PyOperon Pareto prediction statistics. Preserve invalid native outputs
  and validation-selected expressions even when test prediction fails.
- Added budget-timeout native token checkpoints/recovery for DSR and gplearn;
  replay preserves protected prediction semantics and performs no new search.
- Added selective per-workunit repair manifests, baseline hash guards and frozen
  data-hash checks; original results are untouched and valid ODE components are
  referenced read-only in repaired aggregates.
- Verification: focused deep-tree/type/unit/scoring regressions, full repository
  tests and node2 original-failure production-command/native-recovery preflight.
  Exact final counts and job identity will be recorded after verification.
- Scope: repair/re-execution evidence, not MySR results or cross-method superiority.

- Confirmed validation: full Benchmark suite33 passed in4.93s; compileall and
  git diff --check passed; all240 SRSD unit mappings verified,24 angular mappings
  corrected. Four original failure cases passed the actual selective campaign
  command on node2 with matching original data hashes/seed/budget. DSR/gplearn
  native checkpoint replay both produced finite scores without additional search.
- Frozen repair selection:64,019 native fits in1,891 original workunits, covering
  all six methods. Original outputs remain unchanged; the final run identity and
  ODE partial-component gate are recorded in the separate repair run archive.

## 2026-10-14 — TF4SR train-only constant-fit overflow repair

- The live selective run exposed scipy trust-region/SVD failures from extremely
  large finite residuals. Extend the existing `1e10` nonfinite penalty to large
  finite normalized residuals; preserve smaller residuals and retain the original
  seeded decoding, train-only optimization and `max_nfev=200`.
- Exported predictions and shared scores are unchanged by the numerical penalty;
  invalid predictions and poor finite fits remain explicit measured outcomes.
- Verification: full Benchmark suite34 passed in5.77s, including the real scipy
  trust-region regression. The frozen adapter is being checked against all14
  initially observed original SVD failures with unchanged data/seed/budget:
  all14 completed successfully; the complete provenance and saved-prediction
  metric audit also passed all14 with zero issues. The existing256-CPU production
  job continues under its already frozen source.
- Corrective results are preserved separately. Their first-attempt selection
  policy, original-failure hashes and actual resource measurements are explicit;
  final completeness and scientific-quality claims remain deferred.
