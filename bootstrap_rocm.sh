#!/usr/bin/env bash
# Create/update the relocatable FastAFD ROCm environment and install minisgl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.conda-env}"
ENV_FILE="${ENV_FILE:-$ROOT/environment.rocm7.pinned.yml}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CONDA_EXE_VALUE="${CONDA_EXE:-$(command -v conda || true)}"

if [[ ! -x "$ROCM_PATH/bin/hipcc" ]]; then
  echo "hipcc not found: $ROCM_PATH/bin/hipcc" >&2
  echo "Install ROCm 7.x or set ROCM_PATH to the ROCm installation." >&2
  exit 1
fi
if [[ -z "$CONDA_EXE_VALUE" || ! -x "$CONDA_EXE_VALUE" ]]; then
  echo "conda was not found; install Miniforge/Conda or set CONDA_EXE." >&2
  exit 1
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "[bootstrap] updating existing environment: $ENV_PREFIX"
  "$CONDA_EXE_VALUE" env update --prefix "$ENV_PREFIX" --file "$ENV_FILE"
else
  echo "[bootstrap] creating environment: $ENV_PREFIX"
  "$CONDA_EXE_VALUE" env create --prefix "$ENV_PREFIX" --file "$ENV_FILE"
fi

export ROCM_PATH
export PATH="$ENV_PREFIX/bin:$ROCM_PATH/bin:$PATH"
"$ENV_PREFIX/bin/python" -m pip install --no-deps -e "$ROOT"
if [[ -n "${SKIP_RUNTIME_CHECK:-}" ]]; then
  echo "[bootstrap] SKIP_RUNTIME_CHECK is set; live GPU validation deferred"
else
  "$ENV_PREFIX/bin/python" "$ROOT/scripts/check_rocm_runtime.py"
fi

echo "[bootstrap] ready"
echo "ENV_PREFIX=$ENV_PREFIX MODEL=/path/to/model TP=4 GPUS=0,1,2,3 ./run_col_rocm.sh"
