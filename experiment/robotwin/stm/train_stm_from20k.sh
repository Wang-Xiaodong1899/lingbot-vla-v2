#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$root/src"
assets="${STM_ASSETS_ROOT:-$(cd "$root/../../.." && pwd)}"
export STM_ASSETS_ROOT="$assets"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/lingbotvla/bin/python}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$src:$src/lingbotvla/models/vla/vision_models/lingbot-depth:$src/lingbotvla/models/vla/vision_models/MoGe"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LINGBOT_DCP_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$assets/data/train_cache/huggingface" TMPDIR="$assets/data/train_tmp"
export TRITON_CACHE_DIR="$assets/data/triton_train"
export LINGBOT_AUG_LEVEL=0 LINGBOT_STATE_NOISE_STD=0 LINGBOT_STM_GRAD_AUDIT=1
unset LINGBOT_AUG_POLICY LINGBOT_B_AUG LINGBOT_Z4_CONFIG
export OUTPUT_DIR="${OUTPUT_DIR:-$assets/outputs/stm-from20k-gbs512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}" SAVE_INTERVAL_SECONDS="${SAVE_INTERVAL_SECONDS:-7200}"
mkdir -p "$OUTPUT_DIR" "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR"
exec 9>"$OUTPUT_DIR/train.lock"
flock -n 9 || { echo 'Training already holds the output lock.' >&2; exit 1; }
export NGPU="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
"$python_bin" "$root/prepare_training.py"
cd "$src"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    tasks/vla/train_lingbotvla.py "$OUTPUT_DIR/launch/config.yaml" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
