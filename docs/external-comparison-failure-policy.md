# External comparison with incomplete or invalid models

Apply the same rules to MySR and each of the six registered external methods.
Preserve the registered tasks, splits, operators, seeds and search/resource
budgets. Software-only recovery reads candidates generated within the original
budget and performs no additional search. Select candidates by validation data;
freeze that choice before reading test data.

Report two separate outcomes for every fit:

1. Search termination, including normal completion and budget exhaustion.
2. Prediction usability: an expression plus finite training, validation and
   held-out scores. A usable timeout checkpoint contributes its actual error.

A mathematically invalid prediction, incomplete native decoding or budget with
no recoverable candidate is a measured algorithm outcome. Preserve the formula,
frontier and actual nonfinite predictions when available. Such a record does
not acquire a fabricated finite error or a substitute constant model.
Infrastructure faults, corrupt artifacts and implementation exceptions require
repair and explicit audit; they are not algorithmic failures.

For each external-versus-MySR pair, fix the shared applicable task/seed set using
the registry before examining outcomes. Report prediction coverage, search-timeout
rate and the fraction of all registered paired fits achieving each error
threshold. An invalid/no-model fit contributes zero to each threshold numerator
and remains in the denominator. Useful fixed NRMSE thresholds are
`[1e-6, 1e-4, 1e-2, 0.1, 1, 10]`; freeze them before observing MySR results.
Report task families, noise and resource tracks separately before any pooled
statistic. Applicability exclusions must be declared from the registry, not
chosen after a method fails.

Conditional error summaries on usable predictions are supplementary and must
show their numerator and denominator; they alone do not rank the methods.
Do not select the best test score across candidates or retries. Keep the first
corrected attempt chosen by the frozen correction manifest, retaining all prior
attempts and recording the adapter/resource epoch and checkpoint-only cost.

The B6 repair has 394 recoverable checkpoint records among 4,546 unresolved B5
records. Recover these without new search; retain the other 4,152 recorded
negative outcomes instead of repeating unchanged searches. B6 is a corrective
subset, not the final benchmark matrix. Final comparison additionally requires
the complete baseline/repair overlay, ODE aggregation, prediction/provenance
audit and matched MySR results.
