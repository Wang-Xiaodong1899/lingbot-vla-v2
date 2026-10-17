#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
echo "[$(date -Is)] Waiting for all 50 converted tasks"
while ! /root/miniconda3/envs/lingbotvla/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('data/conversion_logs/lerobot_status.json')
try:
    s=json.loads(p.read_text())
    ready=s['completed_tasks']==50 and not s['failed']
except (OSError, ValueError, KeyError):
    ready=False
raise SystemExit(0 if ready else 1)
PY
do
    sleep 20
done
echo "[$(date -Is)] Conversion complete; launching training"
exec bash experiment/robotwin/train_clean_from4k.sh "$@"
