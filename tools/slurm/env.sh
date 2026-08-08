#!/usr/bin/env bash
# Source this after entering the FastAFD container:
#   source tools/slurm/env.sh

export ENV_PREFIX="${ENV_PREFIX:-/opt/afd-env}"
export MODEL="${MODEL:-/scratch/models/gpt-oss-120b}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ENV_PREFIX/bin:$ROCM_PATH/bin:$PATH"

# Slurm/Pyxis exposes the allocated AMD GPU slice through ROCR_VISIBLE_DEVICES.
# Current Ray rejects that name and expects HIP_VISIBLE_DEVICES instead. Preserve
# the exact Slurm-provided slice, but expose it through the variable accepted by
# both Ray and torch. If the caller already chose HIP_VISIBLE_DEVICES, keep it.
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
unset ROCR_VISIBLE_DEVICES

FASTAFD_USER="${FASTAFD_USER:-${USER:-$(id -un)}}"
FASTAFD_CODE="${FASTAFD_CODE:-/scratch/$FASTAFD_USER/FastAFD-MI300X}"
if [[ ! -d "$FASTAFD_CODE" ]]; then
  echo "FastAFD checkout is missing: $FASTAFD_CODE" >&2
  return 1 2>/dev/null || exit 1
fi
cd "$FASTAFD_CODE"

# Make it obvious that this interactive shell is inside the FastAFD container.
# Keep the real compute-node hostname unchanged because Slurm and networking use it.
if [[ $- == *i* ]]; then
  fastafd_prompt_node="${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-gpu}}}"
  PS1="(fastafd:${fastafd_prompt_node}) "'\u:\w\$ '
  unset fastafd_prompt_node
fi

echo "FastAFD environment"
echo "  code : $FASTAFD_CODE"
echo "  env  : $ENV_PREFIX"
echo "  model: $MODEL"
echo "  ROCm : $ROCM_PATH"
