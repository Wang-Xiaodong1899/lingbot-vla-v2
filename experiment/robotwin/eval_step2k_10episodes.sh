#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export QWEN3VL_PATH="$project_dir/evaluation/Qwen3-VL-4B-Instruct"
export VK_ICD_FILENAMES="$project_dir/evaluation/nvidia_icd.json"
export XDG_RUNTIME_DIR="$project_dir/evaluation/runtime"
export TMPDIR="$project_dir/evaluation/tmp"
mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR"
chmod 700 "$XDG_RUNTIME_DIR"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

exec bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$project_dir/lingbot-vla-2-robotwin-clean-bs512-step2k/checkpoints/global_step_2000/hf_ckpt" \
    --inference_workdir "$project_dir/" \
    --eval_workdir "$project_dir/RoboTwin" \
    --output_base "$project_dir/evaluation/results" \
    --conda_sh /root/miniconda3/etc/profile.d/conda.sh \
    --inference_env lingbotvla --sim_env RoboTwin \
    --task_config demo_clean --num_tasks 50 --num_episodes 10 \
    --num_gpus 4 --num_per_gpu 2 \
    --use_bf16 False --use_fp32 True --use_compile True \
    --no_video "$@"
