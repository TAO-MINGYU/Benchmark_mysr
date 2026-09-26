# External solver execution protocol v2

This execution layer replaces the `adapter_pending` launchers. It runs PySR,
Operon, DSR-PyTorch, AI-Feynman, gplearn and TF4SR. MySR is excluded.
The original 315 tasks, four tabular variants, two resource tracks and ten
formal seeds remain unchanged. ODEBench is a separate extension.

## Inputs, selection and retained evidence

Only training observations enter a solver's fit. Candidate selection minimizes
validation MSE, breaking ties by native complexity and then expression text.
The test file is opened after selection; clean and observed test scores and
selected-model predictions are retained. Every solver's exposed frontier or
candidate collection is retained, including invalid candidates. No unavailable
HOF is invented: gplearn exposes retained programs, DSR its valid program cache,
and TF4SR one greedy decoding. `test_numeric_agreement` is agreement on held-out
points, **not** a symbolic-equivalence proof. Symbolic recovery and structural
edit-distance scoring remain separate, uncomputed endpoints in these records.

Every run keeps the input and adapter hashes, immutable request, stdout/stderr,
expressions, frontier, predictions, node/job identity and resource measurements.
Completed requests can be resumed; changed requests at the same output path are
rejected. A timeout, invalid expression or not-applicable case is a retained
result, never silently removed from the matrix. AI-Feynman's root Pareto set is
checkpointed and can be scored after a search timeout without continuing search.

## Budgets and honest comparability

| Track | Search wall time | Native work limit where supported | RSS limit |
|---|---:|---:|---:|
| constrained_resource | 120 seconds | 20,000 | 8 GiB |
| capability_ceiling | 900 seconds | 500,000 | 32 GiB |

A process-group supervisor separates startup (300 seconds), search and scoring
(180 seconds), with 200 ms RSS/CPU sampling. Search termination has a two-second
grace for final native output. Startup/scoring overhead is reported separately
and remains included in total runtime. CPU time uses waited-child resource accounting (reaped descendant work included),
with sampled process CPU also retained as a diagnostic lower bound. RSS limits are sampled enforcement, not kernel cgroup guarantees. Slurm's incorrect
`RealMemory=1 MiB` configuration still prevents a scheduler memory-efficiency claim.
Native solver temporary files are written in a private node-local run directory,
then copied to the durable run archive after the worker exits. Jobs use one CPU with `srun --cpu-bind=cores` and one numerical-library thread.

These runs use a documented **native-method configuration**. They do not claim
an identical mathematical search space or identical evaluation accounting:

- PySR enforces `max_evals` but does not expose its actual counter through the
  installed Python interface; actual evaluations are null with a reason.
- Operon reports native fitness calls. gplearn reports generated programs and
  uses its protected operators. DSR reports sampled programs, with constants
  fitted by its native routine.
- AI-Feynman has no global evaluation counter or global evaluation limit. Its
  wall/RSS limits are enforced; evaluations are null rather than fabricated.
- TF4SR performs one pretrained greedy decode from 50 seeded positive-input
  training rows, using official logarithmic normalization and inverse scaling.
  Its shared constant symbol is fitted on training data only, at most 200 least
  squares function evaluations. This is inference plus constant fitting, not GP.

Native complexity definitions are retained and explicitly named; they are not
interchangeable cross-method tree sizes. `formal_claim=false` means these raw
solver records have not undergone final campaign/scorer/provenance audit. It does
not mean the solver was skipped. Do not use them to claim an overall ranking
until applicability, pretraining overlap and the missing recovery endpoints are
accounted for. No common-space result is implied by these native-method runs.

## Applicability and environment compatibility

AI-Feynman runs on known-expression tasks, including the registered classics and
ODE derivative problems; black-box tasks are not applicable. TF4SR runs only on
SRSD tasks with at most six inputs and at least 50 positive-input, nonzero-target
training rows. It is not applied to ODE or black-box/legacy tasks. Pretraining
contamination has not been ruled out and must accompany TF4SR interpretation.

DSR's source uses removed `collections.Mapping` and NumPy scalar aliases. The
adapter provides equivalent legacy aliases inside its isolated worker. The
previously prepared source has no usable Cython extension; the adapter uses the
upstream Python evaluator when that extension is absent. This affects throughput
and is recorded, rather than pretending the Cython build was validated.

AI-Feynman's console-script bodies are retained, with their shebangs rewritten
in the private run directory to the staged interpreter. Otherwise the Fortran
launchers would silently return to the NFS environment.

The six locked environments are copied, without package upgrades, to private
node-local caches from SHA256-verified tar bundles. The unused Julia registry is
excluded from the offline PySR runtime bundle. Source and environment bundle
hashes accompany deployment; libraries and DSR/TF4SR source are read locally.

## ODE extension

Each system has one parameter setting and two initial conditions. For the first
trajectory, the first 70% of observations form train and the remaining 30% form
validation. Each split is differentiated **independently**, using second-order
finite differences and dropping its two edge points. Training never uses the
reference equation or clean latent derivative. The second clean trajectory is
reserved for derivative-prediction testing. The metric is held-out derivative
prediction, not integrated trajectory rollout or exact vector-field recovery.
Official noise and missing-point conditions remain intact in train/validation.

Each vector-field component is fitted independently. The system's search-time
and native-work budgets are divided equally among its components. Source
integration failures are recorded as data errors. The old condition-level arrays
would contain 1,260 records per element; the replacement arrays use one system
and condition per element (315 elements per method), with 20 system-level runs.
All 63 × 5 × 6 × 2 × 10 = 37,800 system-level slots remain represented, including
TF4SR not-applicable outcomes. ODE totals are not combined with tabular totals.

## Validation and execution

Pilot seeds are now disjoint from the unchanged formal seed list. Old pilot
records are retained with their original seeds; they are not formal results.
`validate.sh` checks real clean/noisy fits, identical-seed repeats and an ODE
component on both compute nodes. `launch.sh` executes a registered work unit and
writes progress after every completed run. Runtime bundles and result data live
outside this code repository. Deployment manifests record replacement job IDs
and the exact code revision.
