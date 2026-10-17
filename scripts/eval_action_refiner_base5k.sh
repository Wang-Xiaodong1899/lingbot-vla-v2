#!/usr/bin/env bash
# Default: evaluate latest saved step, pinned before GPU wait.
# Specific step: bash scripts/eval_action_refiner_base5k.sh --step 1500
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/lingbotvla/bin/python}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON_BIN" experiment/robotwin/eval_saved_action_refiner.py \
    --train-output outputs/action-refiner-base5k-bs768 \
    --expected-base /root/autodl-tmp/models/robotwin-clean-from4k-bs768-step5000 \
    "$@"
