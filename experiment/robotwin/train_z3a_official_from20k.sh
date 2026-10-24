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
# The Z4 wrapper reuses the launch/data checks with a separate policy and output.
export LINGBOT_TRAIN_RECIPE="${LINGBOT_TRAIN_RECIPE:-z3a}"
case "$LINGBOT_TRAIN_RECIPE" in
    z3a)
        unset LINGBOT_INSTRUCTION_POOL
        export LINGBOT_AUG_POLICY=z3a LINGBOT_B_AUG=1 LINGBOT_AUG_LEVEL=1
        export LINGBOT_STATE_NOISE_STD="${STATE_NOISE_STD:-0.01}"
        export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z3a-local-from20k-gbs1024}"
        ;;
    z4)
        unset LINGBOT_INSTRUCTION_POOL
        export LINGBOT_AUG_POLICY=z4 LINGBOT_B_AUG=0 LINGBOT_AUG_LEVEL=0
        export LINGBOT_STATE_NOISE_STD=0
        export LINGBOT_Z4_CONFIG="${LINGBOT_Z4_CONFIG:-$project_dir/configs/augmentation/z4.yaml}"
        export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z4-from20k-gbs1024}"
        ;;
    z6b)
        export LINGBOT_AUG_POLICY=z6b LINGBOT_B_AUG=1 LINGBOT_AUG_LEVEL=5
        export LINGBOT_STATE_NOISE_STD="${STATE_NOISE_STD:-0.01}"
        export LINGBOT_INSTRUCTION_POOL="${LINGBOT_INSTRUCTION_POOL:-$project_dir/assets/training_data/official_instruction_pool_v1.json.gz}"
        export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-z6b-from-pica1300-gbs2048}"
        ;;
    *) echo 'LINGBOT_TRAIN_RECIPE must be z3a, z4 or z6b' >&2; exit 1 ;;
esac
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME" "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/train.lock"
flock -n 9 || { echo "Another launcher holds $OUTPUT_DIR/train.lock" >&2; exit 1; }
num_gpus="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
if (( num_gpus < 1 )); then echo 'No visible CUDA GPUs' >&2; exit 1; fi
export NGPU="$num_gpus"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
if [[ ! "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || [[ ! "$MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( GLOBAL_BATCH_SIZE % (num_gpus * MICRO_BATCH_SIZE) != 0 )); then
    echo 'GLOBAL_BATCH_SIZE must be divisible by visible GPUs × MICRO_BATCH_SIZE (positive integers)' >&2
    exit 1
fi
export GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (num_gpus * MICRO_BATCH_SIZE)))
"$python_bin" tools/prepare_z3a_official.py
if [[ "$LINGBOT_TRAIN_RECIPE" == z4 ]]; then
    export LINGBOT_Z4_CONFIG="$OUTPUT_DIR/launch/z4_augmentation.yaml"
fi
if [[ "$LINGBOT_TRAIN_RECIPE" == z6b ]]; then
    export LINGBOT_INSTRUCTION_POOL="$OUTPUT_DIR/launch/instruction_pool_v1.json.gz"
fi
config="$OUTPUT_DIR/launch/config.yaml"
command=("$python_bin" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node="$num_gpus" tasks/vla/train_lingbotvla.py "$config")
if (( $# )); then
    echo 'Use environment settings (MAX_STEPS, SAVE_STEPS, LR, LR_MIN, MICRO_BATCH_SIZE, OUTPUT_DIR); arbitrary CLI overrides are disabled.' >&2
    exit 1
fi
echo "Muon / policy=$LINGBOT_AUG_POLICY / state noise=$LINGBOT_STATE_NOISE_STD / GPUs=$num_gpus × micro=$MICRO_BATCH_SIZE × accumulation=$GRAD_ACCUM = $GLOBAL_BATCH_SIZE"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 || "${CHECK_ONLY:-0}" == 1 ]]; then
    echo 'Training was not launched.'
    exit 0
fi
"${command[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
