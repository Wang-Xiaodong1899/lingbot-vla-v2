#!/usr/bin/env bash
# Resume the complete bs768 step2000 state; do not use the interrupted step4000.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/robotwin-clean-from4k-bs768-resume2k-save1k}"
export TRAIN_CONFIG="$project_dir/configs/vla/robotwin/robotwin_clean_bs768_resume2k_save500.yaml"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE=768
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"
export DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-1}"
export LINGBOT_DCP_THREADS=1

resume_checkpoint="$(/root/miniconda3/envs/lingbotvla/bin/python - "$project_dir" "$OUTPUT_DIR" "${RESUME_CHECKPOINT:-}" "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
from pathlib import Path
from lingbotvla.checkpoint.integrity import validate_checkpoint
root, output, explicit, devices = sys.argv[1:]
world_size = len(devices.split(','))
if explicit:
    candidates = [Path(explicit).resolve()]
else:
    candidates = sorted((Path(output)/'checkpoints').glob('global_step_[0-9]*'),
                        key=lambda p:int(p.name.split('_')[-1]), reverse=True)
    if not candidates:
        candidates = [Path(root)/'outputs/robotwin-clean-from4k-bs768/checkpoints/global_step_2000']
for p in candidates:
    try:
        validate_checkpoint(p, world_size)
        print(p)
        break
    except Exception as exc:
        print(f'Reject incomplete checkpoint: {p}: {exc}',file=sys.stderr)
else:
    raise SystemExit('No complete resume checkpoint; training was not started')
PY
)"
echo "Resume full training state: $resume_checkpoint"
echo "Output: $OUTPUT_DIR"
echo "Save interval: 500 optimizer steps (next save is step2500 when resuming step2000)"
exec bash experiment/robotwin/train_clean_from4k.sh \
    --train.load_checkpoint_path "$resume_checkpoint" "$@"
