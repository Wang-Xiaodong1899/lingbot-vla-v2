#!/usr/bin/env bash
set -euo pipefail
export NODE_RANK="${1:?Usage: train16_node.sh NODE_RANK}"
P=/home/tiger/workspace/lingbot_vla2_910b_20261012
source "$P/microbatch-probe-20261021/network16.sh"
R="$P/xpolicylab-wyt_dev"
S="$P/temporal-memory-20261021/src"
export PYTHONPATH="$S"
export ASCEND_GLOBAL_LOG_LEVEL=3
cd "$S"
exec "$P/.venv/bin/python" -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" tasks/vla/train_lingbotvla.py "$P/temporal-memory-20261021/train.temporal.yaml"
