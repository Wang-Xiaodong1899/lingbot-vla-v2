#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/lingbotvla/bin/python}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "${NPROC_PER_NODE:-1}" == "1" ]]; then
    exec "$PYTHON_BIN" tasks/vla/train_action_refiner.py "$@"
else
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" tasks/vla/train_action_refiner.py "$@"
fi
