#!/usr/bin/env bash
# Z6b level 5 from explicit PICA1300 HF student weights, with a fresh Muon state.
# This is sequential fine-tuning; PICA distillation is disabled for this stage.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${INIT_MODEL:?Set INIT_MODEL to the PICA global_step_1300 HF checkpoint (not the Z4 step400 descendant)}"
export INIT_MODEL
export LINGBOT_TRAIN_RECIPE=z6b
export NORM_FILE="${NORM_FILE:-$(dirname "$INIT_MODEL")/robotwin_clean_norm.json}"
export ROBOT_CONFIG="${ROBOT_CONFIG:-$project_dir/configs/robot_configs/robotwin.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z6b-from-pica1300-gbs2048}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2048}"
export LR="${LR:-1e-4}" LR_MIN="${LR_MIN:-5e-5}"
export LR_DECAY_STYLE=cosine LR_DECAY_RATIO="${LR_DECAY_RATIO:-2.0}"
export MAX_STEPS="${MAX_STEPS:-500}" SAVE_STEPS="${SAVE_STEPS:-500}"
export DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-2}"
export ENABLE_FP32="${ENABLE_FP32:-1}"
exec bash "$project_dir/experiment/robotwin/train_z3a_official_from20k.sh" "$@"
