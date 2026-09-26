#!/usr/bin/env bash
set -euo pipefail
: "${EXTERNAL_CODE:?immutable benchmark code directory}"
: "${EXTERNAL_BUNDLES:?frozen runtime bundle directory}"
: "${EXTERNAL_ROOT:?comparison source root}"
: "${EXTERNAL_ARCHIVE:?corpus archive}"
: "${EXTERNAL_MANIFEST:?registered campaign manifest}"
: "${EXTERNAL_OUTPUT:?external result directory}"
: "${METHOD:?method id}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export PYTHONPATH="$EXTERNAL_CODE"
case "$METHOD" in
  pysr) prefix=env_1_pysr;;
  operon) prefix=parallel_operon;;
  dsr) prefix=parallel_dsr;;
  ai_feynman_2) prefix=parallel_ai_feynman;;
  gplearn) prefix=parallel_gplearn;;
  tf4sr) prefix=parallel_tf4sr;;
  *) exit 2;;
esac
# System Python is sufficient to stage tar archives, avoiding another NFS environment import.
cache="/tmp/mysr-external-${UID}"
env_root=$(python3 "$EXTERNAL_CODE/benchmark_mysr/external/stage.py" --bundles "$EXTERNAL_BUNDLES" --cache "$cache" --name "$prefix")
sources=$(python3 "$EXTERNAL_CODE/benchmark_mysr/external/stage.py" --bundles "$EXTERNAL_BUNDLES" --cache "$cache" --name sources)
# The supervisor needs psutil. Use the pinned development Python for this small process.
args=()
if [[ "${EXTERNAL_ODE:-0}" == 1 ]]; then args+=(--ode); fi
exec srun --cpu-bind=cores --ntasks=1 "$EXTERNAL_ROOT/../conda_envs/env_1_mysr/bin/python" -m benchmark_mysr.external.campaign \
  --manifest "$EXTERNAL_MANIFEST" --method "$METHOD" --index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "$EXTERNAL_OUTPUT" --archive "$EXTERNAL_ARCHIVE" \
  --sources "$sources" --env-root "$env_root" "${args[@]}"
