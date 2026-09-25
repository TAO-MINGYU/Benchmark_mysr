# Formal comparison protocol

This repository contains the frozen protocol surface for the formal MySR
benchmark. Task membership, source revisions, noise variants, seeds, resources,
methods and failure handling are machine-readable under `manifests/formal/` and
are explained in [`BENCHMARK_USAGE_PLAN.md`](BENCHMARK_USAGE_PLAN.md).

The comparison roster is PySR, Operon, DSR, AI-Feynman 2.0, gplearn, TF4SR, and
MySR. The six baselines run first; MySR runs last after the code-quality window.
Every method has common-space and native-default tracks, and every result retains
its seed, full frontier, metrics, resource data and failure state.

The formal matrix is clean/noisy by constrained/capability-ceiling resource
track. There is no infinite-resource condition. The resource budgets are finite,
frozen before formal runs, and calibrated only on the declared pilot tasks.

Known-expression tasks are scored with exact, structural and numerical
 equivalence metrics. Black-box tasks are scored by held-out prediction,
complexity and resource use; they never receive an exact-recovery claim without
a verified ground-truth expression.
