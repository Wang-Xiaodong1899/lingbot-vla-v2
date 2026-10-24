#!/usr/bin/env bash
# STM runs in an isolated copy of the pinned zhuojianli/icm source snapshot.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STM_ASSETS_ROOT="${STM_ASSETS_ROOT:-$project_dir}"
# Preserve the existing run's launcher and output lock on the training host.
default_work_dir="$project_dir/experiment/robotwin/stm"
if [[ -f /root/autodl-tmp/projects/lingbot-vla-stm/train_stm_from20k.sh ]]; then
    default_work_dir=/root/autodl-tmp/projects/lingbot-vla-stm
fi
exec bash "${STM_WORK_DIR:-$default_work_dir}/train_stm_from20k.sh" "$@"
