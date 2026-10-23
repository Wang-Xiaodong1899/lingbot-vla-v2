#!/usr/bin/env bash
# Sequential fine-tuning: PICA student HF weights -> Z4, with a fresh Muon state.
# Download first: python tools/download_pica_z4_model.py
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export INIT_MODEL="${INIT_MODEL:-/root/autodl-tmp/models/Xiaodong-lingbotvla-robotwin-clean-sft-20k-PICA/global_step_2000}"
export NORM_FILE="${NORM_FILE:-$(dirname "$INIT_MODEL")/robotwin_clean_norm.json}"
# The HF repository contains normalization but no robot mapping YAML.
# This is the same 14-D joint/camera mapping used by the verified 20k baseline.
export ROBOT_CONFIG="${ROBOT_CONFIG:-$project_dir/configs/robot_configs/robotwin.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z4-from-pica2000-gbs1024}"
export LR="${LR:-1e-5}"
export LR_MIN="${LR_MIN:-5e-6}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
exec bash "$project_dir/experiment/robotwin/train_z4_official_from20k.sh" "$@"
