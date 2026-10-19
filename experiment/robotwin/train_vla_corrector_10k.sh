#!/usr/bin/env bash
# Full extraction -> corrector training -> real OGG smoke -> queued clean/random evaluation.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$project_dir/data/lerobot_v21_source/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXTRACT_CUDA_VISIBLE_DEVICES="${EXTRACT_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
extract_num_gpus="$(awk -F, '{print NF}' <<< "$EXTRACT_CUDA_VISIBLE_DEVICES")"
python_bin=/root/miniconda3/envs/lingbotvla/bin/python
base=/root/autodl-fs/lingbotvla-robotwin-clean-sft/global_step_10000
output="${OUTPUT_DIR:-$project_dir/outputs/vla-corrector-base10k}"
cache="${CACHE_DIR:-$project_dir/data/corrector_features_10k}"
mkdir -p "$output"
exec 9>"$output/pipeline.lock"
flock -n 9 || { echo "Corrector pipeline already running"; exit 1; }
num_gpus="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
common=(--base-model "$base" --qwen-path "$project_dir/evaluation/Qwen3-VL-4B-Instruct" \
 --norm-stats "$base/robotwin_clean_norm.json" --train-list "$project_dir/assets/training_data/robotwin_clean50_local.txt")
status() { "$python_bin" - "$output" "$1" "$$" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,os
p=Path(sys.argv[1])/'pipeline_status.json'
t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=sys.argv[2],pid=int(sys.argv[3]),gpus=os.environ['EXTRACT_CUDA_VISIBLE_DEVICES'] if sys.argv[2]=='extracting_features' else os.environ['CUDA_VISIBLE_DEVICES'],training_gpus=os.environ['CUDA_VISIBLE_DEVICES'],updated_at=datetime.now(timezone.utc).isoformat())));os.replace(t,p)
PY
}
trap 'status failed' ERR
if [[ "${SKIP_EXTRACTION:-0}" == "1" ]]; then
 "$python_bin" - "$cache" <<'PY_CHECK'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);m=json.loads((root/'metadata.json').read_text());files=list(root.glob('task*/complete.json'))
assert not m.get('smoke_max_episodes') and len(files)==m['total_episodes'], 'Extraction incomplete'
assert sum(json.loads(p.read_text())['frames'] for p in files)==m['total_frames'], 'Frame count mismatch'
print('Complete visual cache verified; proceeding directly to training',flush=True)
PY_CHECK
else
status extracting_features
echo "Extract full frozen 10k spatial features on GPUs $EXTRACT_CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES="$EXTRACT_CUDA_VISIBLE_DEVICES" "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$extract_num_gpus" \
 tasks/vla/extract_corrector_features.py "${common[@]}" --output "$cache" \
 --batch-size "${EXTRACT_BATCH_SIZE:-1024}" 2>&1 | tee -a "$output/extract.log"
fi
status training_corrector
resume_args=()
if [[ -f "$output/train/latest.pt" ]]; then
 resume_args=(--resume "$output/train/latest.pt")
fi
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$num_gpus" \
 tasks/vla/train_vla_corrector.py --cache "$cache" --output "$output/train" \
 --batch-size "${MICRO_BATCH_SIZE:-128}" --epochs "${EPOCHS:-30}" \
 --workers "${DATA_NUM_WORKERS:-4}" --save-every 500 "${resume_args[@]}" \
 2>&1 | tee -a "$output/train.log"
status checking_real_ogg
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$python_bin" tools/check_vla_corrector_batch.py \
 --checkpoint "$output/train/best.pt" --cache "$cache" --output "$output/real_ogg_smoke.json" \
 2>&1 | tee -a "$output/real_ogg_smoke.log"
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
 status waiting_for_evaluation_gpus
 "$python_bin" experiment/robotwin/eval_vla_corrector_10k.py \
 --checkpoint "$output/train/best.pt" --output "$project_dir/evaluation/results/vla_corrector_10k" \
 --gpus "${EVAL_GPUS:-0,1,2,3}" --episodes 10 --tasks 50 \
 2>&1 | tee -a "$output/eval_driver.log"
fi
status complete
