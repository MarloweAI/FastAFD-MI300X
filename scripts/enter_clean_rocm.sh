#!/usr/bin/env bash
# ROCm/MI300X counterpart to enter_clean.sh.
#
# Same contract as the CUDA version: `env -i` blanks the environment and
# `--noprofile --norc` skips all rc files, so kernel builds are reproducible.
# CONDA_SH / CACHE_ROOT / WORKDIR are interpolated by the PARENT shell and so
# survive the wipe; everything else does not.
#
# Differences from scripts/enter_clean.sh (all CUDA-specific there):
#   - CUDA_HOME / CUDA_PATH / CUDA_NVCC_EXECUTABLE  -> ROCM_PATH / HIP_PATH.
#     There is no nvcc; tvm_ffi invokes hipcc from $ROCM_PATH/bin itself.
#   - TRITON_PTXAS_BLACKWELL_PATH                   -> dropped. No ptxas on AMD;
#     ROCm Triton emits GCN ISA directly.
#   - $CONDA_PREFIX/targets/{sbsa,x86_64}-linux/lib -> dropped. That layout is
#     the CUDA toolkit's; ROCm libs live in $ROCM_PATH/lib.
#   - The site-packages nvidia/nccl/lib probe               -> dropped. RCCL comes
#     from $ROCM_PATH/lib (librccl.so), not from a pip wheel.
#   - FLASHINFER_WORKSPACE_BASE                     -> dropped. FlashInfer has no
#     ROCm build; this port replaces its ops (dev_log/qwen/03_port_plan.md M1.4).
#   - MINISGL_DEEPEP_BUILD_DIR / MINISGL_DEEPGEMM_BUILD_DIR / EP_JIT_CACHE_DIR /
#     N2M_M2N_GIN_BUILD_DIR are still exported so the dirs exist, but DeepEP and
#     DeepGEMM do not build on gfx942 at all (dev_log/qwen/02_dependency_inventory.md
#     §D, §E). They are here only so a stray import fails on its own terms
#     rather than on a missing path.
#   - Exports MINISGL_QK_NORM_ROPE_BACKEND=off, required until the wave64 port of
#     qk_norm_rope.cu lands.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  enter_clean_rocm.sh CONDA_ENV_NAME

Example:
  ./scripts/enter_clean_rocm.sh minisgl-rocm7

Requires CONDA_SH to point at your own conda (a personal miniforge prefix on a
shared host). Source ~/.env-<name>.sh first if it is not already set.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

ENV_NAME="$1"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
START_DIR="${PWD}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-${START_DIR}/cache}"
ROCM_PATH_VALUE="${ROCM_PATH:-/opt/rocm}"
USER_NAME="${USER:-$(id -un)}"
LOGNAME_VALUE="${LOGNAME:-$USER_NAME}"
TERM_VALUE="${TERM:-xterm-256color}"
SHELL_VALUE="${SHELL:-/bin/bash}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  echo "hint: source ~/.env-<name>.sh first, or export CONDA_SH explicitly." >&2
  exit 1
fi

if [[ ! -x "${ROCM_PATH_VALUE}/bin/hipcc" ]]; then
  echo "hipcc not found under ROCM_PATH=${ROCM_PATH_VALUE}" >&2
  exit 1
fi

exec env -i \
  HOME="${HOME}" \
  USER="${USER_NAME}" \
  LOGNAME="${LOGNAME_VALUE}" \
  TERM="${TERM_VALUE}" \
  SHELL="${SHELL_VALUE}" \
  PATH="/usr/bin:/bin" \
  LANG="C.UTF-8" \
  LC_ALL="C.UTF-8" \
  BASH_ENV="" \
  ENV="" \
  PROMPT_COMMAND="" \
  bash --noprofile --norc -lc "
    set -eo pipefail
    cd \"${WORKDIR}\"
    source \"${CONDA_SH}\"
    conda activate \"${ENV_NAME}\"
    mkdir -p \"${CACHE_ROOT}/tvm-ffi\"
    mkdir -p \"${CACHE_ROOT}/deepep-jit\"
    mkdir -p \"${CACHE_ROOT}/gin-comm\"
    mkdir -p \"${CACHE_ROOT}/deepep-moe\"
    mkdir -p \"${CACHE_ROOT}/deepgemm\"
    export ROCM_PATH=\"${ROCM_PATH_VALUE}\"
    export HIP_PATH=\"${ROCM_PATH_VALUE}\"
    export HIP_PLATFORM=amd
    export PATH=\"\${CONDA_PREFIX}/bin:${ROCM_PATH_VALUE}/bin:\${PATH}\"
    export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:${ROCM_PATH_VALUE}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
    export LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:${ROCM_PATH_VALUE}/lib\${LIBRARY_PATH:+:\${LIBRARY_PATH}}\"
    export TVM_FFI_CACHE_DIR=\"${CACHE_ROOT}/tvm-ffi\"
    export EP_JIT_CACHE_DIR=\"${CACHE_ROOT}/deepep-jit\"
    export N2M_M2N_GIN_BUILD_DIR=\"${CACHE_ROOT}/gin-comm\"
    export MINISGL_DEEPEP_BUILD_DIR=\"${CACHE_ROOT}/deepep-moe\"
    export MINISGL_DEEPGEMM_BUILD_DIR=\"${CACHE_ROOT}/deepgemm\"
    export MINISGL_QK_NORM_ROPE_BACKEND=off
    export ENV_NAME=\"${ENV_NAME}\"
    echo \"[clean-shell] env=\${CONDA_DEFAULT_ENV:-} prefix=\${CONDA_PREFIX:-}\"
    echo \"[clean-shell] rocm=\${ROCM_PATH} hipcc=\$(command -v hipcc || echo '<missing>')\"
    # NOTE: no \`grep -m1\` here — it exits early, SIGPIPEs rocminfo, and
    # \`set -o pipefail\` then reports the pipeline as failed even though the
    # match succeeded, firing the fallback branch spuriously.
    __gpu_arch=\$( { rocminfo 2>/dev/null || true; } | grep -o 'gfx[0-9a-z]*' | head -n1 )
    echo \"[clean-shell] gpu=\${__gpu_arch:-<rocminfo unavailable>}\"
    echo \"[clean-shell] TVM_FFI_CACHE_DIR=\${TVM_FFI_CACHE_DIR} MINISGL_QK_NORM_ROPE_BACKEND=\${MINISGL_QK_NORM_ROPE_BACKEND}\"
    exec bash --noprofile --norc -i
  "
