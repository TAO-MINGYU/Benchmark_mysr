# External comparison protocol v2

This is a new complete six-method experiment. Earlier results remain available
as v1 evidence and must not be combined with v2 scores or used to choose the
best test outcome. MySR will be run later against this same frozen protocol.
The machine-readable contract is `manifests/formal/fair-comparison-v2.json`.

The registered matrix contains 315 tabular tasks, four noise variants, two
resource tracks and ten seeds, plus 63 ODE systems in five conditions with the
same tracks and seeds. There are 189,000 top-level slots and 221,400 native
scalar-target fits: 151,200 tabular fits and 70,200 ODE component fits. Each
external method receives all 36,900 scalar fits. Native applicability exclusions
remain explicit records; failures never become exclusions.

## Common computation and data

Each scalar target gets one hardware thread on a distinct physical CPU core,
with 120 seconds of search in the constrained track or 900 seconds in the
capability track. These are wall-clock search budgets, not CPU-second budgets.
ODE derivatives are fitted independently: each component receives a full scalar
budget and a D-dimensional system receives D times that search allocation.
Native evaluation counts have incompatible meanings and are recorded rather
than used as a common stopping limit. Implementation-required count ceilings
are nonbinding sentinels. Finite search iteration sentinels remain documented in
the adapter; exhaustion is visible as an early native return.

Imports, synthetic PySR compilation, expression export, checkpoint replay and
held-out scoring are measured separately. Data-dependent neural fitting,
constant optimization and decoding consume the search budget. Recovery only
scores actual native checkpoints and never performs a new search. Slurm has no
wall-time cap, memory cap or automatic requeue; subprocess startup and scoring
have no extra wall-time or memory cap. The 120/900-second scientific search
budgets are intentional experiment constraints. Available-memory admission
checks defer new launches; they do not limit a running fit's memory.

Every solver receives the same observed training split. Compute max-absolute
input and target scales from that split only, without centering; zero scales
become one, and publicly specified radian columns keep scale one. Invert these
multiplicative transforms for exported expressions and all predictions. This
avoids many overflows caused by physical-unit magnitudes without reading clean
training targets or validation/test statistics. Native complexity excludes this
shared preprocessing and is not a universal expression-size measure.

Use validation MSE, then native complexity and expression text, to select one
model. Read test inputs/targets only after selection is frozen. Preserve the
selected expression even if its test predictions are invalid; never select a
second candidate after inspecting test scores. Retain native frontiers,
checkpoints, actual predictions (including nonfinite values), split hashes,
source hashes, environments, seed, search/total elapsed time, process-tree peak
RSS and child CPU accounting. Peak RSS is sampled; it is not Slurm memory
accounting or a guaranteed peak measurement.

## Native method settings and limitations

This compares native methods under common computation. It does not assert
identical operator semantics or search spaces.

| Method | Frozen choice |
| --- | --- |
| PySR | Float64, 31 populations of 27, maxsize 30, serial deterministic search; ordinary arithmetic and sin/cos/exp/log/sqrt. Replay the native HOF CSV without costly symbolic rewrites. |
| Operon | Population 1,000, MSE/length objectives, LM coefficient optimization for 10 iterations, one thread. Native build scalar type remains an environment property. Preserve completed-generation Pareto trees at timeout. |
| DSR | CPU, batch 100, ordinary add/sub/mul/div/sin/cos/exp/log/sqrt and constants, GP meld disabled. Preserve sampled native programs and constants. |
| gplearn | Population 1,000, MSE, one worker; native protected div/log/sqrt, sin/cos and an ordinary NumPy exp registered through the native function API. Preserve retained native programs. |
| AI-Feynman | Native 14ops grammar, degree-3 polynomial fit, 100 neural epochs, no internal held-out test split. Brute-force attempt time is derived from the overall search track. Numerical gradient guards preserve regular native scoring. Original known-expression applicability registry retained. |
| TF4SR | Frozen pretrained model on CPU. Official 50 positive-input training rows, at most six variables and native geometric scaling. One greedy decode with masks for absent inputs, special tokens and impossible tree completion; no beam search or added training. Native positive shared C fitted on training data with max_nfev=200. |

TF4SR's pretraining overlap with evaluation formulas is unknown. Report it as a
separate pretrained stratum; do not claim unseen-formula generalization. Protected
operators, pretrained/fixed grammars and method-specific input requirements must
be disclosed alongside common-compute results. No reference formula is passed to
any solver; ground-truth availability is only public registry metadata.

The installed PySR wheel's direct-URL commit is the runtime provenance. A
separate upstream checkout can differ and must not be cited as the executed
version without verification. Other package versions, source commits, bundle
checksums, native precision and single-CPU affinity must accompany the run.

## Coverage, negative results and MySR

Invalid models and no-model budget outcomes are part of the denominator for all
registered applicable fits. Report prediction coverage and attainment at the
predeclared NRMSE thresholds 1e-6, 1e-4, 1e-2, 0.1, 1 and 10. Conditional finite
error summaries require their denominator. Raw undefined errors stay null;
there are no fabricated expressions, infinities in JSON scores or silent drops.
Implementation/runtime faults require review rather than being reclassified as
mathematical failures.

Report family, noise level, resource track and method separately. ODE summaries
require every component, use macro-average component NRMSE on the independent
clean trajectory and retain all component evidence. Sum CPU time and serial
component elapsed time without presenting the latter as concurrent system wall
time. The automatic post-run audit checks identities, data/source/writer hashes,
frontier and prediction checksums, saved-prediction metrics, expected record
counts and complete ODE summaries. Audit failure blocks comparison readiness.

Later MySR must use these exact snapshots, seeds, training-only scaling, Float64,
scalar component units, single physical CPU, search budgets, validation selection
and negative-outcome rules. Its requested public vocabulary is the same; record
any native features/semantics and training provenance, and use no reference
formula, held-out information or task-specific parameter tuning. External-only
completion is not a completed MySR comparison or publication evidence.

## Reproduction

`python -m benchmark_mysr.external.fair_plan` expands the original full manifests
into a new data snapshot, line-indexed `requests.jsonl`, `fits-index.json` and a
count/checksum summary. It refuses to overwrite a previous preparation.
`python -m benchmark_mysr.external.fair` verifies a request line and protocol
hash, checks its frozen budget and requires one assigned CPU before native fit.
The node pool verifies complete request/index hashes and runs independent fits
on distinct physical cores. `python -m benchmark_mysr.external.fair_finalize`
performs the final audit and ODE aggregation. Preserve the run configuration,
submission receipt, Slurm allocation, native validation and source release
beside the results. Never mutate a live release or restart production to inspect
its progress.
