#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/data/lerobot_v21_source/src:$project_dir/lingbotvla/models/vla/vision_models/lingbot-depth:$project_dir/lingbotvla/models/vla/vision_models/MoGe:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# PICA uses its own paired visual augmentation recipe.
export LINGBOT_B_AUG=0
export LINGBOT_STATE_NOISE_STD=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LINGBOT_DCP_THREADS="${LINGBOT_DCP_THREADS:-1}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$project_dir/data/train_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$project_dir/data/train_tmp"
export TORCHINDUCTOR_CACHE_DIR="$project_dir/data/torchinductor_train"
export TRITON_CACHE_DIR="$project_dir/data/triton_train"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

train_config="${TRAIN_CONFIG:-$project_dir/configs/vla/robotwin/robotwin_pica_from20k.yaml}"
train_output_dir="${OUTPUT_DIR:-$project_dir/outputs/robotwin-pica-from20k}"
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
num_gpus="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
if (( global_batch_size % (micro_batch_size * num_gpus) != 0 )); then
    echo "Global batch must be divisible by GPUs x micro batch"
    exit 1
fi
accumulation_steps=$((global_batch_size / (micro_batch_size * num_gpus)))

python tools/check_pica_training.py "$train_config"

mkdir -p "$train_output_dir"
exec 9>"$train_output_dir/train.lock"
flock -n 9 || { echo "A training launcher already holds $train_output_dir/train.lock"; exit 1; }
echo "Global batch $global_batch_size, LR ${LR:-1e-5}, save model + optimizer + training state every 500 steps"
echo "GPUs=$num_gpus micro_batch=$micro_batch_size accumulation=$accumulation_steps"
echo "Training log: $train_output_dir/train.log"
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "CHECK_ONLY: preflight complete; training was not launched"
    exit 0
fi
python tools/monitor_training_resources.py "$train_output_dir/resource_memory.jsonl" "$$" 9>&- &
resource_monitor_pid=$!
trap 'kill "$resource_monitor_pid" 2>/dev/null || true' EXIT
torchrun --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    tasks/vla/train_lingbotvla.py "$train_config" \
    --train.output_dir "$train_output_dir" \
    --train.data_parallel_shard_size "$num_gpus" \
    --train.micro_batch_size "$micro_batch_size" \
    --train.gradient_accumulation_steps "$accumulation_steps" \
    --train.global_batch_size "$global_batch_size" \
    --train.lr "${LR:-1e-5}" --train.lr_min "${LR_MIN:-5e-6}" \
    --data.num_workers "${DATA_NUM_WORKERS:-4}" --data.prefetch_factor "${DATA_PREFETCH_FACTOR:-1}" \
    --train.save_steps 500 --train.save_hf_weights true \
    --train.async_save_hf_weights false \
    --train.enable_resume true \
    --train.max_steps "${MAX_STEPS:-30000}" "$@" \
    2>&1 | tee -a "$train_output_dir/train.log"
