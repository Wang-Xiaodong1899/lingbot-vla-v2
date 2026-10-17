#!/usr/bin/env bash
# Launch manually: bash experiment/robotwin/run_action_refiner_10k_mb96.sh
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MICRO_BATCH_SIZE=192
export GRAD_ACCUM=1
export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/action-refiner-base10k-bs768-mb192-acc1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
echo "4 GPUs x micro batch 192 x accumulation 1 = global batch 768"
echo "Save every 500 optimizer steps; normalization: base checkpoint robotwin_clean_norm.json"
echo "Output: $OUTPUT_DIR"
bash experiment/robotwin/train_action_refiner_10k.sh "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
