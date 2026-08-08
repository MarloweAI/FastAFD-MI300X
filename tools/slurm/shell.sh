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
SHARED_IMAGE="${FASTAFD_NFS_IMAGE:-/nfs/containers/sqsh/$(basename "$IMAGE")}"
CODE_DIR="${FASTAFD_CODE:-/scratch/$FASTAFD_USER/FastAFD-MI300X}"
MODEL_DIR="${FASTAFD_MODEL:-/scratch/models/gpt-oss-120b}"
ENV_FILE="$TOOLS_DIR/env.sh"
CONTAINER_BASHRC="$TOOLS_DIR/container.bashrc"
TARGET_NODE="${FASTAFD_NODE:-}"
PARTITION="${FASTAFD_PARTITION:-gpu}"

if ! [[ "$GPUS" =~ ^[1-8]$ ]]; then
  echo "GPU count must be an integer from 1 through 8." >&2
  exit 2
fi
if ! [[ "$TIME_LIMIT" =~ ^[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "Time must have the form HH:MM:SS (for example 06:00:00)." >&2
  exit 2
fi

# /scratch is local to each compute node. When a node is requested explicitly,
# first verify that setup has seeded its source and model. Then ensure the
# immutable image is present before Pyxis tries to open it. Copy through a
# partial file so an interrupted transfer is never runnable.
if [[ -n "$TARGET_NODE" ]]; then
  echo "Checking FastAFD setup on $TARGET_NODE."
  srun \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=0 \
    --mem=2G \
    --cpus-per-task=2 \
    --time=00:30:00 \
    --job-name=stage-fastafd-image \
    --nodelist="$TARGET_NODE" \
    bash -s -- "$SHARED_IMAGE" "$IMAGE" "$CODE_DIR" "$MODEL_DIR" "$TOOLS_DIR/setup.sh" <<'STAGE_IMAGE'
set -euo pipefail
src=$1
dst=$2
code=$3
model=$4
setup=$5

missing=()
[[ -d "$code/.git" ]] || missing+=("source checkout: $code")
for name in config.json tokenizer.json model.safetensors.index.json; do
  [[ -s "$model/$name" ]] || missing+=("model file: $model/$name")
done
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: FastAFD setup is incomplete on $(hostname):" >&2
  printf '  missing %s\n' "${missing[@]}" >&2
  echo "Prepare this node once from the head node:" >&2
  echo "  FASTAFD_NODE=${SLURMD_NODENAME:-$HOSTNAME} $setup" >&2
  exit 4
fi

if [[ ! -f "$src" ]]; then
  echo "ERROR: shared container image does not exist: $src" >&2
  echo "Run tools/slurm/setup.sh to build it first." >&2
  exit 1
fi

src_size=$(stat -c %s "$src")
if [[ -f "$dst" && "$(stat -c %s "$dst")" == "$src_size" ]]; then
  echo "Container image is already staged: $dst"
  exit 0
fi

echo "Container image is missing on $(hostname); staging it from NFS."
echo "  source: $src"
echo "  target: $dst"
mkdir -p "$(dirname "$dst")"
tmp="${dst}.partial.$$"
trap 'rm -f "$tmp"' EXIT
cp "$src" "$tmp"
chmod 644 "$tmp"
[[ "$(stat -c %s "$tmp")" == "$src_size" ]] || {
  echo "ERROR: staged image size does not match the shared image." >&2
  exit 1
}
mv -f "$tmp" "$dst"
trap - EXIT
echo "Container image staging complete: $dst"
STAGE_IMAGE
  echo
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
echo "Model:     $MODEL_DIR"
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
