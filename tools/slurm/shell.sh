#!/usr/bin/env bash
# Reserve GPUs and enter the already-staged FastAFD container.
#
#   ./shell.sh             # 4 GPUs for 6 hours
#   ./shell.sh 1           # 1 GPU for 6 hours
#   ./shell.sh 8 08:00:00  # 8 GPUs for 8 hours
#   FASTAFD_NODE=mi300x-02 ./shell.sh 4  # use one prepared node
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
TOOLS_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
FASTAFD_USER="${FASTAFD_USER:-${USER:-$(id -un)}}"
GPUS="${1:-4}"
TIME_LIMIT="${2:-06:00:00}"
IMAGE="${FASTAFD_IMAGE:-/scratch/images/fastafd-rocm724-v1.sqsh}"
CODE_DIR="${FASTAFD_CODE:-/scratch/$FASTAFD_USER/FastAFD-MI300X}"
ENV_FILE="$TOOLS_DIR/env.sh"
CONTAINER_BASHRC="$TOOLS_DIR/container.bashrc"
TARGET_NODE="${FASTAFD_NODE:-}"

if ! [[ "$GPUS" =~ ^[1-8]$ ]]; then
  echo "GPU count must be an integer from 1 through 8." >&2
  exit 2
fi
if ! [[ "$TIME_LIMIT" =~ ^[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "Time must have the form HH:MM:SS (for example 06:00:00)." >&2
  exit 2
fi

cmd=(
  srun
  --nodes=1
  --gpus="$GPUS"
  --time="$TIME_LIMIT"
  --job-name=fastafd-dev
)
if [[ -n "$TARGET_NODE" ]]; then
  cmd+=(--nodelist="$TARGET_NODE")
fi
cmd+=(
  --container-image="$IMAGE"
  --container-workdir="$CODE_DIR"
  --pty
  bash --rcfile "$CONTAINER_BASHRC" -i
)

echo "Requesting $GPUS GPU(s) for $TIME_LIMIT."
echo "Container: $IMAGE"
echo "Code:      $CODE_DIR"
echo "Node:      ${TARGET_NODE:-any prepared GPU node}"
echo
echo "Slurm command:"
printf '  %q' "${cmd[@]}"
printf '\n\n'
echo "After the container prompt appears, run:"
echo "  source $ENV_FILE"
echo
echo "Exit the container shell to gracefully stop background jobs and release the allocation."
echo

exec "${cmd[@]}"
