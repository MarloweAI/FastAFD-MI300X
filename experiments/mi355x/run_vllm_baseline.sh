#!/usr/bin/env bash
set -euo pipefail

node=${SLURM_NODELIST:-}
hardware_access=${HARDWARE_ACCESS_CHECKOUT:-/workspace/$USER/hardware-access}
inferencex=${INFERENCEX_CHECKOUT:-/workspace/$USER/inferencex-770268c}
run_label=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)-vllm}

[[ -x $hardware_access/sites/colovore/health-tests/inference/scripts/run_gptoss120b_inferencex.sh ]] || {
  echo "missing hardware-access baseline runner: $hardware_access" >&2; exit 1;
}
if [[ -n $node ]]; then
  export SLURM_NODELIST=$node
else
  unset SLURM_NODELIST
fi
export INFERENCEX_CHECKOUT=$inferencex RUN_LABEL=$run_label
exec "$hardware_access/sites/colovore/health-tests/inference/scripts/run_gptoss120b_inferencex.sh"
