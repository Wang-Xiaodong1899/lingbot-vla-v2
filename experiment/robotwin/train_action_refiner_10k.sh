#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/data/lerobot_v21_source/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
output_dir="${OUTPUT_DIR:-$project_dir/outputs/action-refiner-base10k-bs768-mb96-acc2}"
mkdir -p "$output_dir"
exec 9>"$output_dir/train.lock"
flock -n 9 || { echo "Training already running in $output_dir"; exit 1; }
bash scripts/train_action_refiner.sh \
 --base-model /root/autodl-fs/lingbotvla-robotwin-clean-sft/global_step_10000 \
 --qwen-path "$project_dir/evaluation/Qwen3-VL-4B-Instruct" \
 --train-list "$project_dir/assets/training_data/robotwin_clean50_local.txt" \
 --norm-stats /root/autodl-fs/lingbotvla-robotwin-clean-sft/global_step_10000/robotwin_clean_norm.json \
 --output-dir "$output_dir" --base-dtype float32 \
 --batch-size "${MICRO_BATCH_SIZE:-96}" --grad-accum "${GRAD_ACCUM:-2}" \
 --workers "${DATA_NUM_WORKERS:-0}" --save-every 500 "$@"
