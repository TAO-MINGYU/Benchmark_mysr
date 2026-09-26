#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
export PYTHONPATH="$EXTERNAL_CODE"
methods=(pysr operon dsr ai_feynman_2 gplearn tf4sr)
prefixes=(env_1_pysr parallel_operon parallel_dsr parallel_ai_feynman parallel_gplearn parallel_tf4sr)
export METHOD="${methods[$SLURM_ARRAY_TASK_ID]}"
prefix="${prefixes[$SLURM_ARRAY_TASK_ID]}"
cache="/tmp/mysr-external-${UID}"
export STAGED_ENV_ROOT=$(python3 "$EXTERNAL_CODE/benchmark_mysr/external/stage.py" --bundles "$EXTERNAL_BUNDLES" --cache "$cache" --name "$prefix")
export STAGED_SOURCES=$(python3 "$EXTERNAL_CODE/benchmark_mysr/external/stage.py" --bundles "$EXTERNAL_BUNDLES" --cache "$cache" --name sources)
srun --cpu-bind=cores --ntasks=1 "$EXTERNAL_ARCHIVE/../../conda_envs/env_1_mysr/bin/python" "$EXTERNAL_CODE/scripts/external/validate.py"
