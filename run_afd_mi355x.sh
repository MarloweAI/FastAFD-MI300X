#!/usr/bin/env bash
# Launch one full-node MI355X AFD split: A TP1 attention workers + B TP1 FFN workers.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:?MODEL must point to the local GPT-OSS-120B snapshot}
ATTN_GPUS=${ATTN_GPUS:-4}
FFN_GPUS=${FFN_GPUS:-4}
PORT=${PORT:-19297}
MEM_RATIO=${MEM_RATIO:-0.82}
GRAPH_MAX_BS=${GRAPH_MAX_BS:-128}
NUM_MB=${NUM_MB:-2}
# Continuous admission above eight active requests currently wedges the AFD
# collective schedule after a partially drained batch. Keep the external client
# concurrency unrestricted and queue above the validated active-set size.
MAX_RUNNING_REQ=${MAX_RUNNING_REQ:-8}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-aiter}

for name in ATTN_GPUS FFN_GPUS GRAPH_MAX_BS NUM_MB MAX_RUNNING_REQ; do
  value=${!name}
  [[ "$value" =~ ^[0-9]+$ ]] || { printf '%s must be an integer, got %q\n' "$name" "$value" >&2; exit 2; }
done
(( ATTN_GPUS >= 1 && FFN_GPUS >= 1 )) || { echo "both role counts must be positive" >&2; exit 2; }
(( ATTN_GPUS + FFN_GPUS == 8 )) || {
  echo "MI355X Pareto runs require a full-node split summing to 8 GPUs" >&2
  exit 2
}
[[ "$MOE_RUNNER_BACKEND" == aiter || "$MOE_RUNNER_BACKEND" == triton ]] || {
  echo "MOE_RUNNER_BACKEND must be aiter or triton" >&2
  exit 2
}

if [[ -n ${ROCR_VISIBLE_DEVICES:-} && -z ${HIP_VISIBLE_DEVICES:-} ]]; then
  export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
fi
unset ROCR_VISIBLE_DEVICES

export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export MINISGL_MXFP4_PACKED=1
export MINISGL_AFD_MOE_BACKEND=rccl
export MINISGL_AITER_MAX_M=${MINISGL_AITER_MAX_M:-4096}
export MINISGL_M2N_FIXED_SHAPE=${MINISGL_M2N_FIXED_SHAPE:-1}
export MINISGL_PYNCCL_MAX_BUFFER_SIZE=${MINISGL_PYNCCL_MAX_BUFFER_SIZE:-0}
export MINISGL_QK_NORM_ROPE_BACKEND=${MINISGL_QK_NORM_ROPE_BACKEND:-off}
export AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS:-0}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$REPO/cache/tvm-ffi-mi355x}
mkdir -p "$TVM_FFI_CACHE_DIR"

python3 "$REPO/scripts/check_mi355x_runtime.py"
printf '[fastafd-mi355x] split=%s:%s ep=%s runner=%s graph=%s num_mb=%s port=%s\n' \
  "$ATTN_GPUS" "$FFN_GPUS" "$FFN_GPUS" "$MOE_RUNNER_BACKEND" \
  "$GRAPH_MAX_BS" "$NUM_MB" "$PORT"

exec python3 -m minisgl \
  --mode afd-serve \
  --cache-type naive \
  --ray-address local \
  --model-path "$MODEL" \
  --afd-attn-dp-size "$ATTN_GPUS" \
  --afd-mlp-dp-size "$FFN_GPUS" \
  --afd-attn-tp-size 1 \
  --afd-mlp-tp-size 1 \
  --afd-mlp-ep-size "$FFN_GPUS" \
  --afd-moe-a2a-backend none \
  --afd-moe-runner-backend "$MOE_RUNNER_BACKEND" \
  --memory-ratio "$MEM_RATIO" \
  --cuda-graph-max-bs "$GRAPH_MAX_BS" \
  --afd-num-mb "$NUM_MB" \
  --afd-max-running-requests "$MAX_RUNNING_REQ" \
  --port "$PORT"
