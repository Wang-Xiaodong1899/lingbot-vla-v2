#!/usr/bin/env bash
# Z4: 10% original observations, 90% combined online image corruptions.
set -euo pipefail
export LINGBOT_TRAIN_RECIPE=z4
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_z3a_official_from20k.sh" "$@"
