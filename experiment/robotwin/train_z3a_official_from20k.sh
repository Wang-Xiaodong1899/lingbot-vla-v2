#!/usr/bin/env bash
# Local adaptation: official LeRobot v3 + aug1 + native Muon, initialized at 20k.
# Z3a's exact upstream config is not in the cached experiments revision ffbf35b.
# DRY_RUN=1: write config and print command. CHECK_ONLY=1: validate without training.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir:$project_dir/lingbotvla/models/vla/vision_models/lingbot-depth:$project_dir/lingbotvla/models/vla/vision_models/MoGe"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/lingbotvla/bin/python}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LINGBOT_DCP_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$project_dir/data/train_cache/huggingface"
export TMPDIR="$project_dir/data/train_tmp"
export TORCHINDUCTOR_CACHE_DIR="$project_dir/data/torchinductor_train"
export TRITON_CACHE_DIR="$project_dir/data/triton_train"
# Set explicitly so another recipe's environment cannot silently disable aug1.
export LINGBOT_B_AUG=1 LINGBOT_AUG_LEVEL=1
export LINGBOT_STATE_NOISE_STD="${STATE_NOISE_STD:-0.01}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z3a-local-from20k-gbs1024}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/train.lock"
flock -n 9 || { echo "Another launcher holds $OUTPUT_DIR/train.lock" >&2; exit 1; }
num_gpus="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
if (( num_gpus < 1 )); then echo 'No visible CUDA GPUs' >&2; exit 1; fi
export NGPU="$num_gpus"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
export GLOBAL_BATCH_SIZE=1024
if [[ ! "$MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( 1024 % (num_gpus * MICRO_BATCH_SIZE) != 0 )); then
    echo '1024 must be divisible by visible GPUs × MICRO_BATCH_SIZE (positive integer)' >&2
    exit 1
fi
export GRAD_ACCUM=$((1024 / (num_gpus * MICRO_BATCH_SIZE)))
"$python_bin" tools/prepare_z3a_official.py
config="$OUTPUT_DIR/launch/config.yaml"
command=("$python_bin" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node="$num_gpus" tasks/vla/train_lingbotvla.py "$config")
if (( $# )); then
    echo 'Use environment settings (MAX_STEPS, SAVE_STEPS, LR, LR_MIN, MICRO_BATCH_SIZE, OUTPUT_DIR); arbitrary CLI overrides are disabled.' >&2
    exit 1
fi
echo "Muon / aug1 / state noise=$LINGBOT_STATE_NOISE_STD / GPUs=$num_gpus × micro=$MICRO_BATCH_SIZE × accumulation=$GRAD_ACCUM = 1024"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 || "${CHECK_ONLY:-0}" == 1 ]]; then
    echo 'Training was not launched.'
    exit 0
fi
"${command[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
