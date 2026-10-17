#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/data/lerobot_v21_source/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
exec /root/miniconda3/envs/lingbotvla/bin/python -m deploy.lingbot_action_refiner_policy \
 --model_path /root/autodl-fs/lingbotvla-robotwin-clean-sft/global_step_10000 \
 --refiner_checkpoint "${REFINER_CHECKPOINT:-$project_dir/outputs/action-refiner-base10k-bs768/latest.pt}" \
 --use_bf16 False --use_fp32 True --use_compile False --use_length 50 --port "${PORT:-9530}" "$@"
