#!/usr/bin/env bash
# Launch AFD (attention-FFN disaggregation) on a single 8-GPU MI300X node.
#
#   ./run_afd_rocm.sh                          # 4 attention + 4 FFN, port 19297
#   ATTN_TP=2 MLP_TP=2 GPUS=0,1,2,3 ./run_afd_rocm.sh   # 2A+2F on cards 0-3
#   PORT=19300 ./run_afd_rocm.sh
#
# GPUS restricts which cards are used (shared host: leave the rest for coworkers).
# It sets HIP_VISIBLE_DEVICES, so torch/Ray see only that slice and rank N binds to
# the Nth listed card.
#
# Then:  PORT=19297 ./ask_rocm.sh "Capital of France?"
#
# Why each setting (see dev_log/qwen/12_afd_wireup.md):
#   --mode afd-serve                 the AFD runtime rather than the colocated server
#   --cache-type naive               the AFD centralized scheduler rejects the radix cache
#   --ray-address local              AFD needs Ray; this self-hosts instead of requiring
#                                    an external `ray start --head`
#   --cuda-graph-max-bs $GRAPH_MAX_BS  0 (eager) for now. AFD runs eager not because the
#                                    attention backend lacks capture -- triton_decode has it
#                                    (dev_log/16) -- but because the M2N transport uses
#                                    data-dependent shapes that capture forbids. That costs
#                                    AFD roughly the 4.1x graphs gave colocated; see
#                                    dev_log/qwen/17_remaining_items.md sec 1.
#   MINISGL_AFD_MOE_BACKEND=rccl     M2N transport over torch.distributed collectives;
#                                    DeepEP cannot build on gfx942 (NCCL device API +
#                                    SM90 TMA). Defaulted on ROCm, set here explicitly.
#   --afd-moe-runner-backend triton  DeepGEMM is SM90/SM100-only; the Triton BF16
#                                    expert path is bridged by moe/m2n_permute.py
#   MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 torch's bundled librccl.so lacks
#                                    ncclCommWindowRegister (the system one has it, but
#                                    same SONAME so torch's wins). 0 skips the
#                                    symmetric-memory window, which is an optimization.
#   MINISGL_QK_NORM_ROPE_BACKEND=off qk_norm_rope.cu is not wave64-ported yet
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/miniforge3-yanxiong/envs/minisgl-rocm7}"
MODEL="${MODEL:-/home/marlowe/models/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-19297}"
ATTN_TP="${ATTN_TP:-4}"
MLP_TP="${MLP_TP:-4}"
MLP_EP="${MLP_EP:-$MLP_TP}"
# Attention ranks size their KV pool from memory_ratio BEFORE loading weights, so the
# 0.9 default leaves no room and OOMs mid-load (dev_log/qwen/12_afd_wireup.md sec 5).
MEM_RATIO="${MEM_RATIO:-0.5}"
# 0 = eager. AFD graph capture reaches the transport and then dies on
# `flat_ids[valid]` in rccl_m2n_adapter.py:220 -- boolean-mask indexing is illegal
# while a stream is capturing. Making the M2N transport fixed-shape is what unlocks
# this, and it is specified in dev_log/qwen/17_remaining_items.md sec 1.5. Set
# GRAPH_MAX_BS=8 to reproduce the failure while working on that.
GRAPH_MAX_BS="${GRAPH_MAX_BS:-0}"
# Micro-batch count == the AFD pipeline depth, and it is the whole point of AFD.
#
# At NUM_MB=1 the schedule in afd_attention_worker.py is strictly SERIAL: dispatch layer L,
# then block on the EG rank finishing layer L before computing L+1. The attention ranks idle
# through every expert GEMM and the expert ranks idle through every attention -- 4 cards doing
# the work of 2, alternately, plus transport latency. No transport optimisation can make that
# configuration beat colocated, because the loss is structural.
#
# At NUM_MB>=2 the same schedule interleaves (C(L,mb0), compute+D(L+1,mb0), C(L,mb1), ...) so
# mb1's expert work overlaps mb0's attention. That is the pipeline AFD needs to pay off.
#
# BUT IT IS MEASURABLY WORSE TODAY, so the default stays 1 (doc 26 §5):
#     decode B=32, ctx 8192:  NUM_MB=1 166.3 ms   NUM_MB=2 306.6 ms   (0.54x)
#     prefill 8192:           NUM_MB=1 5931.9     NUM_MB=2 5876.4     (1.01x, noise)
# Micro-batching splits the batch but DOUBLES the number of dispatch/combine round trips
# per layer, and each round trip costs ~4 ms of host-sync orchestration (doc 26 §3). So it
# doubles the dominant cost while halving only the part that was never the bottleneck.
#
# The pipeline cannot pay off until the transport stops serialising on the host. Flip this
# to 2 AFTER the fixed-shape M2N work, and re-measure -- that is the ordering the numbers
# force, and it is why "turn on micro-batching" is not a standalone fix.
NUM_MB="${NUM_MB:-1}"

# Max concurrent requests. The default came from ServerArgs.afd_batch_size=8 and this script
# never exposed it, so AFD was capped at 8 running requests by an unset default -- not by
# hardware or design. That cap is why the AFD-vs-colocated THROUGHPUT gap (25-59x) is so much
# worse than the per-request ITL gap (11-12x): above B=8 AFD serves queued groups of 8 while
# colocated serves all B. And AFD's step is flat in batch (76-92 ms from B=1 to B=64), so
# serving more should be close to free. See dev_log/gpt_oss_120b/28 §20 and §21.
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-64}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ENV_PREFIX/bin:$ROCM_PATH/bin:$PATH"
export TVM_FFI_CACHE_DIR="$REPO/cache/tvm-ffi"
export MINISGL_QK_NORM_ROPE_BACKEND=off
if [[ -n "${GPUS:-}" ]]; then
  # HIP_VISIBLE_DEVICES is the ROCm equivalent of CUDA_VISIBLE_DEVICES; Ray and torch
  # both honour it, so ranks are numbered within this slice.
  #
  # Do NOT also set ROCR_VISIBLE_DEVICES: the two **compose** instead of aliasing.
  # ROCR filters first, then HIP indexes into the survivors, so GPUS=4 gave ROCR one
  # card and HIP a request for index 4 of it -> "No HIP GPUs are available". A 0-based
  # contiguous prefix like 0,1,2,3 hid the bug by making the second filter an identity
  # map. See dev_log/qwen/14_performance.md sec 8.
  export HIP_VISIBLE_DEVICES="$GPUS"
elif [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  # Slurm/Pyxis supplies the allocated slice through ROCR_VISIBLE_DEVICES, but
  # current Ray requires HIP_VISIBLE_DEVICES. Keep Slurm's exact value.
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
unset ROCR_VISIBLE_DEVICES
export MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-rccl}"
export MINISGL_PYNCCL_MAX_BUFFER_SIZE="${MINISGL_PYNCCL_MAX_BUFFER_SIZE:-0}"

# Pin gloo to the real NIC. The M2N counts exchange is a gloo (CPU) all_to_all_single, so
# an unroutable interface choice hangs every forward. This host carries k8s/docker
# interfaces (flannel.1, cni0, docker0, plus eth1) alongside eth0, and gloo's auto-select
# can land on one that does not route between Ray actors: the symptom is
# "Timed out waiting 300000ms for send operation to complete" with EVERY rank parked in
# the same all_to_all, not a connection error. Default to the interface owning the default
# route, which is the one Ray reports as the node IP.
IFACE_DEFAULT=$(ip -o -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${IFACE_DEFAULT:-eth0}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
mkdir -p "$TVM_FFI_CACHE_DIR"

for f in "$ENV_PREFIX/bin/python" "$ROCM_PATH/bin/hipcc"; do
  [[ -x "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -e "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

for value_name in ATTN_TP MLP_TP MLP_EP; do
  value=${!value_name}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $value_name must be a positive integer; got $value_name=$value" >&2
    exit 2
  fi
done
TOTAL=$(( ATTN_TP + MLP_TP ))
NGPU=$("$ENV_PREFIX/bin/python" -c \
  'import torch; print(torch.cuda.device_count())' 2>/dev/null) || {
  echo "ERROR: could not determine the GPUs visible inside this container." >&2
  exit 1
}
if ! [[ "$NGPU" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid visible GPU count: $NGPU" >&2
  exit 1
fi
if (( TOTAL > NGPU )); then
  echo "ERROR: ${ATTN_TP} attention + ${MLP_TP} FFN requires $TOTAL GPUs," >&2
  echo "but this Slurm container exposes only $NGPU. Exit and request at least $TOTAL GPUs." >&2
  exit 1
fi

# Fail early and clearly if someone else is using the GPUs -- this is a shared host.
# Only inspect the cards this run will actually use.
WANT="${GPUS:-all}"
BUSY=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Used Memory" \
        | awk -v want="$WANT" '{
             idx = NR - 1
             use = (want == "all")
             if (!use) { n = split(want, a, ","); for (i = 1; i <= n; i++) if (a[i]+0 == idx) use = 1 }
             if (use && $NF/1073741824 > 2) c++
           } END { print c+0 }')
if (( BUSY > 0 )); then
  echo "[run_afd] WARNING: $BUSY of the requested GPU(s) already hold >2 GiB:" >&2
  rocm-smi --showpids 2>/dev/null | awk 'NF>=4 && $3+0 > 0 {printf "           pid=%s %s %.0f GiB\n", $1, $2, $4/1073741824}' >&2
  echo "[run_afd] AFD needs $TOTAL free GPUs. Coordinate before continuing; not killing anything." >&2
fi

echo "[run_afd] ${ATTN_TP} attention + ${MLP_TP} FFN GPUs (ep=${MLP_EP}) on port $PORT"
echo "[run_afd] m2n=$MINISGL_AFD_MOE_BACKEND  runner=triton  memory_ratio=$MEM_RATIO  num_mb=$NUM_MB  graph_max_bs=$GRAPH_MAX_BS"
echo "[run_afd] max_running_req=$MAX_RUNNING_REQ (was capped at 8 by an unset default)"
echo "[run_afd] gloo_iface=$GLOO_SOCKET_IFNAME (M2N counts exchange runs on this; unpinned it can hang)"
echo "[run_afd] model=$MODEL"
echo "[run_afd] NOTE: expect this to be SLOWER than colocated TP8 -- see dev_log/qwen/12_afd_wireup.md"

exec python -m minisgl \
  --mode afd-serve \
  --cache-type naive \
  --ray-address local \
  --model-path "$MODEL" \
  --afd-attn-tp-size "$ATTN_TP" \
  --afd-mlp-tp-size "$MLP_TP" \
  --afd-mlp-ep-size "$MLP_EP" \
  --afd-moe-runner-backend triton \
  --memory-ratio "$MEM_RATIO" \
  --cuda-graph-max-bs "$GRAPH_MAX_BS" \
  --afd-num-mb "$NUM_MB" \
  --afd-max-running-requests "$MAX_RUNNING_REQ" \
  --port "$PORT"
