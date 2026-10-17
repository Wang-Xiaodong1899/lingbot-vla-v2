#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/data/lerobot_v21_source/src:$project_dir/lingbotvla/models/vla/vision_models/lingbot-depth:$project_dir/lingbotvla/models/vla/vision_models/MoGe:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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

train_config="${TRAIN_CONFIG:-$project_dir/configs/vla/robotwin/robotwin_clean_from4k_bs768.yaml}"
train_output_dir="${OUTPUT_DIR:-$project_dir/outputs/robotwin-clean-from4k-bs768}"
micro_batch_size="${MICRO_BATCH_SIZE:-48}"
global_batch_size="${GLOBAL_BATCH_SIZE:-768}"
num_gpus="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
if (( global_batch_size % (micro_batch_size * num_gpus) != 0 )); then
    echo "Global batch must be divisible by GPUs x micro batch"
    exit 1
fi
accumulation_steps=$((global_batch_size / (micro_batch_size * num_gpus)))

# Only publish fully converted datasets to training. This guard prevents
# accidentally training on a partially populated task list.
python - "$project_dir" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
manifest=root/'assets/training_data/robotwin_clean50_local.txt'
lines=[line for line in manifest.read_text().splitlines() if line.strip()]
if len(lines)!=50:
    raise SystemExit(f'Data conversion is not complete: {len(lines)}/50 tasks. Wait for conversion to finish.')
paths=[]
for line in lines:
    robot,path=line.split(maxsplit=1)
    dataset=Path(path)
    marker=json.loads((dataset/'.conversion_complete.json').read_text())
    info=json.loads((dataset/'meta/info.json').read_text())
    assert robot=='robotwin' and marker['episodes']==info['total_episodes']==50, path
    assert info['codebase_version']=='v2.1', path
    paths.append(path)
assert len(set(paths))==50
print('Training data ready: 50 tasks, 2500 episodes',flush=True)
PY

mkdir -p "$train_output_dir"
exec 9>"$train_output_dir/train.lock"
flock -n 9 || { echo "A training launcher already holds $train_output_dir/train.lock"; exit 1; }
echo "Global batch $global_batch_size, LR 1e-4, save model + optimizer + training state every 500 steps"
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
    --train.lr 0.0001 --train.lr_min 0.00005 \
    --data.num_workers "${DATA_NUM_WORKERS:-8}" --data.prefetch_factor "${DATA_PREFETCH_FACTOR:-1}" \
    --train.save_steps 500 --train.save_hf_weights true \
    --train.async_save_hf_weights false \
    --train.enable_resume true \
    --train.max_steps "${MAX_STEPS:-50000}" "$@" \
    2>&1 | tee -a "$train_output_dir/train.log"
