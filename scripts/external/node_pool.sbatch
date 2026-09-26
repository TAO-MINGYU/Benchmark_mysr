#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=512
#SBATCH --threads-per-core=2
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${POOL_CODE:?frozen scheduling code}"
: "${POOL_CONFIG:?immutable deployment configuration}"
: "${POOL_NODE:?registered node}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
exec srun --ntasks=1 --cpus-per-task=512 --cpu-bind=threads \
  /usr/bin/python3 "$POOL_CODE/scripts/external/node_pool.py" \
  --config "$POOL_CONFIG" --node "$POOL_NODE" --workers 512
