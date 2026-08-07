#!/usr/bin/env bash
# Start minisgl on AMD MI300X (gfx942) with the working ROCm configuration.
#
#   ./run_col_rocm.sh                 # BF16 Qwen3-30B-A3B, TP1, port 19295
#   TP=8 ./run_col_rocm.sh            # all 8 GPUs (tensor parallel, colocated)
#   PORT=19300 ./run_col_rocm.sh      # different port
#   GRAPH_MAX_BS=0 ./run_col_rocm.sh  # disable HIP graphs (debugging / A-B)
#   MODEL=/path/to/model ./run_col_rocm.sh
#
# TP > 1 uses pynccl (ported to RCCL in this branch) rather than torch.distributed,
# because torch's process-group watchdog makes HIP graph capture impossible -- see the
# note at the DISABLE_PYNCCL branch below and dev_log/qwen/19_colocated_tp4_perf.md.
#
# NOTE: this is colocated tensor parallelism, NOT AFD. Attention-FFN
# disaggregation (e.g. 4 attention + 4 FFN GPUs) cannot run on gfx942 at all --
# it requires DeepEP, which is built on the NCCL 2.28 device API plus SM90
# TMA/mbarrier. See dev_log/qwen/02_dependency_inventory.md sec E.
#
# Then, from another shell:
#   ./ask_rocm.sh "What is the capital of France?"
#
# Why each setting is required (see dev_log/qwen/STATUS.md):
#   --cuda-graph-max-bs 32           `auto` resolves to torch_ref (prefill) +
#                                    triton_decode (decode); the decode half is
#                                    capturable. Worth 4.1x at TP1 short context
#                                    (dev_log/16) and 4.0x at TP4 / ISL 8192
#                                    (dev_log/19). GRAPH_MAX_BS=0 goes back to eager.
#   MINISGL_QK_NORM_ROPE_BACKEND=off qk_norm_rope.cu is not ported to wave64 yet
#   TVM_FFI_CACHE_DIR                keeps JIT builds in the repo cache/ (gitignored);
#                                    concurrent writes to a shared cache corrupt it
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/miniforge3-yanxiong/envs/minisgl-rocm7}"
MODEL="${MODEL:-/home/marlowe/models/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-19295}"
TP="${TP:-1}"
# Capture up to this decode batch. 32 covers the useful range at ~0.7 GiB; the
# upstream default (160/256) captures ~25 sizes and costs proportionally more.
# HIP graphs are the single largest win here: 4.1x at TP1 short context (dev_log/16)
# and 4.0x at TP4 / ISL 8192 (dev_log/19). They work at every TP -- but only over
# pynccl, see the collectives note below.
GRAPH_MAX_BS="${GRAPH_MAX_BS:-32}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ENV_PREFIX/bin:$ROCM_PATH/bin:$PATH"
export MINISGL_QK_NORM_ROPE_BACKEND=off
if [[ -n "${GPUS:-}" ]]; then
  # Confine the run to a slice of a shared host (same knob as run_afd_rocm.sh).
  # Set ONLY HIP_VISIBLE_DEVICES. These two variables **compose** rather than being
  # aliases: ROCR filters at the runtime level first, then HIP indexes into what
  # survived. Setting both to "4" means ROCR exposes one card (now index 0) and HIP
  # then asks for index 4 of one card -> "No HIP GPUs are available". It happened to
  # work for GPUS=0,1,2,3 only because a 0-based contiguous prefix makes the second
  # filter an identity map. See dev_log/qwen/14_performance.md sec 8.
  export HIP_VISIBLE_DEVICES="$GPUS"
  unset ROCR_VISIBLE_DEVICES
fi
export TVM_FFI_CACHE_DIR="$REPO/cache/tvm-ffi"
mkdir -p "$TVM_FFI_CACHE_DIR"

for f in "$ENV_PREFIX/bin/python" "$ROCM_PATH/bin/hipcc"; do
  [[ -x "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -e "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

echo "[run_rocm] env=$ENV_PREFIX"
echo "[run_rocm] model=$MODEL  tp=$TP  port=$PORT"
echo "[run_rocm] gpu=$( { rocminfo 2>/dev/null || true; } | grep -o 'gfx[0-9a-z]*' | head -n1)"
echo "[run_rocm] first start JIT-compiles a few kernels (~1-3 min); later starts are ~40 s"

EXTRA=()
if (( TP > 1 )); then
  # torch.distributed "nccl" backend == RCCL on ROCm; the vendored pynccl
  # extension does not build here (see header note).
  # Use pynccl (this port's RCCL wrapper), NOT torch.distributed, for TP collectives.
  # torch's process-group watchdog thread queries HIP events recorded on the collective
  # stream; during graph capture those are "captured" events and querying them aborts
  # capture with `hipErrorCapturedEvent`. TORCH_NCCL_ENABLE_MONITORING=0,
  # TORCH_NCCL_ASYNC_ERROR_HANDLING=0 and TORCH_NCCL_TRACE_BUFFER_SIZE=0 all fail to
  # suppress it. pynccl has no watchdog, so capture succeeds -- worth 4.0x at TP4 /
  # ISL 8192. MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 skips the symmetric-memory window,
  # which torch's bundled librccl.so cannot register (dev_log/qwen/CHECKPOINT.md E2).
  # Set DISABLE_PYNCCL=1 to fall back, which also forces graphs off.
  if [[ -n "${DISABLE_PYNCCL:-}" ]]; then
    EXTRA+=(--disable-pynccl)
    GRAPH_MAX_BS=0
    echo "[run_rocm] TP=$TP -> --disable-pynccl; graphs disabled (torch PG breaks capture)"
  else
    export MINISGL_PYNCCL_MAX_BUFFER_SIZE="${MINISGL_PYNCCL_MAX_BUFFER_SIZE:-0}"
    echo "[run_rocm] TP=$TP -> pynccl collectives (required for HIP graph capture)"
  fi
fi

# EXTRA_ARGS: verbatim passthrough for one-off experiments (e.g. --memory-ratio when isolating
# how KV page count affects graph capture, dev_log/gpt_oss_120b/55). Deliberately last so it can
# override anything set above.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(${EXTRA_ARGS})
  echo "[run_rocm] EXTRA_ARGS: ${EXTRA_ARGS}"
fi

exec python -m minisgl \
  --model-path "$MODEL" \
  --tp-size "$TP" \
  --cuda-graph-max-bs "$GRAPH_MAX_BS" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --port "$PORT" \
  "${EXTRA[@]}"
