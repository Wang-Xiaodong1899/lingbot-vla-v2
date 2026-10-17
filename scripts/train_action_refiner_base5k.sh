#!/usr/bin/env bash
# Fresh run: bash scripts/train_action_refiner_base5k.sh
# Resume:    bash scripts/train_action_refiner_base5k.sh --resume outputs/action-refiner-base5k-bs768/latest.pt
# Check:     NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train_action_refiner_base5k.sh --batch-size 1 --check-batch
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
exec bash scripts/train_action_refiner.sh \
    --base-model /root/autodl-tmp/models/robotwin-clean-from4k-bs768-step5000 \
    --output-dir outputs/action-refiner-base5k-bs768 \
    --base-dtype float32 \
    --batch-size 192 \
    --grad-accum 1 \
    --save-every 500 \
    "$@"
