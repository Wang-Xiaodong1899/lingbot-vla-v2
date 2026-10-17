#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CHUNKTRUST_ROOT="${CHUNKTRUST_ROOT:-/root/autodl-tmp/projects/ChunkTrust}"
export QWEN3VL_PATH="$project_dir/evaluation/Qwen3-VL-4B-Instruct"
export VK_ICD_FILENAMES="$project_dir/evaluation/nvidia_icd.json"
export XDG_RUNTIME_DIR="$project_dir/evaluation/runtime"
export TMPDIR="$project_dir/evaluation/tmp"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR"
chmod 700 "$XDG_RUNTIME_DIR"
for setting in ${SETTINGS:-demo_clean demo_randomized}; do
    resume_args=()
    if [ "$setting" = demo_clean ] && [ -n "${RESUME_CLEAN_FROM:-}" ]; then
        resume_args=(--resume_from "$RESUME_CLEAN_FROM")
    fi
    bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
        --model_path "$project_dir/weights/lingbotvla-robotwin-clean-sft/global_step_10000" \
        --inference_script deploy/lingbot_vla_chunktrust_policy.py \
        --inference_workdir "$project_dir/" --eval_workdir "$project_dir/RoboTwin" \
        --output_base "${OUTPUT_BASE:-$project_dir/evaluation/results/chunktrust_10k}" \
        --conda_sh /root/miniconda3/etc/profile.d/conda.sh \
        --inference_env lingbotvla --sim_env RoboTwin \
        --task_config "$setting" --num_tasks "${NUM_TASKS:-50}" \
        --num_episodes "${NUM_EPISODES:-10}" \
        --num_gpus "${NUM_GPUS:-4}" --num_per_gpu "${NUM_PER_GPU:-2}" \
        --start_port "${START_PORT:-9430}" \
        --use_bf16 False --use_fp32 True --use_compile "${USE_COMPILE:-True}" \
        --no_video "${resume_args[@]}" "$@"
done
