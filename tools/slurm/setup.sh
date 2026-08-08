#!/usr/bin/env bash
# One-time FastAFD setup. Run this on the head node.
#
# This script deliberately prints each major Slurm command. It does not overwrite
# an existing image or an existing source checkout.
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
TOOLS_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
REPO_ROOT=$(cd "$TOOLS_DIR/../.." && pwd)
FASTAFD_USER="${FASTAFD_USER:-${USER:-$(id -un)}}"
SOURCE_TREE="${FASTAFD_SOURCE_TREE:-$REPO_ROOT}"
BASE_IMAGE="${FASTAFD_BASE_IMAGE:-docker://rocm/dev-ubuntu-24.04:7.2.4-complete}"
IMAGE_NAME="${FASTAFD_IMAGE_NAME:-fastafd-rocm724-v1.sqsh}"
NFS_IMAGE="${FASTAFD_NFS_IMAGE:-/nfs/containers/sqsh/$IMAGE_NAME}"
LOCAL_IMAGE="${FASTAFD_LOCAL_IMAGE:-/scratch/images/$IMAGE_NAME}"
CODE_DIR="${FASTAFD_CODE:-/scratch/$FASTAFD_USER/FastAFD-MI300X}"
MODEL_REPO="${FASTAFD_MODEL_REPO:-openai/gpt-oss-120b}"
MODEL_DIR="${FASTAFD_MODEL:-/scratch/models/gpt-oss-120b}"
ENV_PREFIX="${FASTAFD_ENV_PREFIX:-/opt/afd-env}"
PARTITION="${FASTAFD_PARTITION:-gpu}"
TARGET_NODE="${FASTAFD_NODE:-}"
SETUP_TIME="${FASTAFD_SETUP_TIME:-12:00:00}"
SLURM_TEMPLATES="${SLURM_TEMPLATES:-/nfs/home/$FASTAFD_USER/slurm/scripts/templates}"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command is missing on the head node: $1" >&2
    exit 1
  }
}

show_command() {
  printf '  %q' "$@"
  printf '\n'
}

require srun
require salloc
require scontrol
require sinfo

if [[ ! -d "$SOURCE_TREE/.git" ]]; then
  echo "Management checkout is missing: $SOURCE_TREE" >&2
  echo "Clone FastAFD there before running setup." >&2
  exit 1
fi

requested_node_args=()
if [[ -n "$TARGET_NODE" ]]; then
  if ! sinfo -h -N -p "$PARTITION" -n "$TARGET_NODE" -o '%N' | grep -qx "$TARGET_NODE"; then
    echo "Node is not in partition $PARTITION: $TARGET_NODE" >&2
    exit 1
  fi
  requested_node_args=(--nodelist="$TARGET_NODE")
fi

# Hold one GPU from beginning to end. Without this outer allocation, each srun
# below becomes an independent job and setup can stall between phases as other
# users consume the node. The setup script itself stays on the head node; its
# srun job steps execute on the one reserved compute node.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  reserve_cmd=(
    salloc
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --gpus=1
    --cpus-per-task=20
    --mem=220G
    --time="$SETUP_TIME"
    --job-name=setup-fastafd
    "${requested_node_args[@]}"
    "$TOOLS_DIR/setup.sh"
  )
  echo "Reserving one GPU for the entire setup with:"
  show_command "${reserve_cmd[@]}"
  echo
  exec "${reserve_cmd[@]}"
fi

mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#allocated_nodes[@]}" -ne 1 ]]; then
  echo "setup.sh requires a one-node allocation; got: $SLURM_JOB_NODELIST" >&2
  exit 1
fi
ALLOCATED_NODE=${allocated_nodes[0]}
if [[ -n "$TARGET_NODE" && "$TARGET_NODE" != "$ALLOCATED_NODE" ]]; then
  echo "Requested node $TARGET_NODE, but allocation is on $ALLOCATED_NODE." >&2
  exit 1
fi
TARGET_NODE=$ALLOCATED_NODE
NODE_COUNT=1
node_args=(--nodelist="$TARGET_NODE")
validation_node_args=(--nodelist="$TARGET_NODE")

echo "FastAFD setup"
echo "  source tree : $SOURCE_TREE"
echo "  base image  : $BASE_IMAGE"
echo "  saved image : $NFS_IMAGE"
echo "  local image : $LOCAL_IMAGE"
echo "  code        : $CODE_DIR"
echo "  model       : $MODEL_DIR"
echo "  GPU nodes   : $NODE_COUNT"
echo "  allocation  : $SLURM_JOB_ID (one GPU held for all setup phases)"
echo "  target node : $TARGET_NODE"
echo "  validation  : $TARGET_NODE"
echo

if [[ -e "$NFS_IMAGE" ]]; then
  echo "Saved image already exists; not rebuilding it: $NFS_IMAGE"
else
  build_cmd=(
    srun
    --partition="$PARTITION"
    --nodes=1
    --gpus=0
    --mem=128G
    --cpus-per-task=8
    --time=06:00:00
    --job-name=build-fastafd
    "${node_args[@]}"
    --container-image="$BASE_IMAGE"
    --container-writable
    --container-remap-root
    --container-save="$NFS_IMAGE"
    bash -s -- "$SOURCE_TREE" "$CODE_DIR" "$ENV_PREFIX"
  )

  echo "Building the image with:"
  show_command "${build_cmd[@]}"
  echo

  "${build_cmd[@]}" <<'INSIDE'
set -euxo pipefail
SOURCE_TREE=$1
CODE_DIR=$2
ENV_PREFIX=$3

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  bc build-essential ca-certificates curl git iproute2 libdw1t64 rsync vim

if [[ ! -x /opt/miniforge/bin/conda ]]; then
  curl -fsSL \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/Miniforge3.sh
  bash /tmp/Miniforge3.sh -b -p /opt/miniforge
  rm -f /tmp/Miniforge3.sh
fi

if [[ -e "$CODE_DIR" && ! -d "$CODE_DIR/.git" ]]; then
  echo "Refusing to replace non-Git path: $CODE_DIR" >&2
  exit 1
fi
if [[ ! -d "$CODE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$CODE_DIR")"
  rsync -a --exclude='/.conda-env/' --exclude='/cache/' --exclude='/models/' \
    --exclude='/results/' "$SOURCE_TREE/" "$CODE_DIR/"
fi
if [[ ! -x "$CODE_DIR/run_col_rocm.sh" ]]; then
  echo "Existing checkout predates run_col_rocm.sh: $CODE_DIR" >&2
  echo "Commit/push the launcher rename, then update this checkout explicitly." >&2
  exit 1
fi

cd "$CODE_DIR"
CONDA_EXE=/opt/miniforge/bin/conda \
ENV_PREFIX="$ENV_PREFIX" \
ROCM_PATH=/opt/rocm \
SKIP_RUNTIME_CHECK=1 \
./bootstrap_rocm.sh

apt-get clean
rm -rf /var/lib/apt/lists/*
INSIDE

  # The head node can briefly cache the earlier "does not exist" lookup.
  for _ in $(seq 1 15); do
    ls "$(dirname "$NFS_IMAGE")" >/dev/null 2>&1 || true
    [[ -e "$NFS_IMAGE" ]] && break
    sleep 2
  done
  if [[ ! -e "$NFS_IMAGE" ]]; then
    echo "The build completed but the image is not visible: $NFS_IMAGE" >&2
    exit 1
  fi
  chmod 644 "$NFS_IMAGE"
fi

if [[ -n "$TARGET_NODE" ]]; then
  echo
  echo "Staging the image to $TARGET_NODE only:"
  stage_cmd=(
    srun
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --gpus=0
    --mem=4G
    --cpus-per-task=2
    --time=00:30:00
    --job-name=stage-fastafd-image
    "${node_args[@]}"
    bash -c
    'set -euxo pipefail
     SRC=$1
     DST=$2
     mkdir -p "$(dirname "$DST")"
     if [[ -f "$DST" ]] && [[ "$(stat -c %s "$DST")" == "$(stat -c %s "$SRC")" ]]; then
       echo "Already staged: $DST"
       exit 0
     fi
     tmp="$DST.partial.$$"
     cp "$SRC" "$tmp"
     chmod 644 "$tmp"
     mv -f "$tmp" "$DST"'
    _ "$NFS_IMAGE" "$LOCAL_IMAGE"
  )
  show_command "${stage_cmd[@]}"
  "${stage_cmd[@]}"
elif [[ -x "$SLURM_TEMPLATES/stage-image.sh" ]]; then
  echo
  echo "Staging the image to every GPU node:"
  show_command "$SLURM_TEMPLATES/stage-image.sh" "$IMAGE_NAME"
  "$SLURM_TEMPLATES/stage-image.sh" "$IMAGE_NAME"
else
  echo "Missing image staging helper: $SLURM_TEMPLATES/stage-image.sh" >&2
  exit 1
fi

echo
echo "Seeding the source checkout on $TARGET_NODE (existing checkouts are untouched)."
source_cmd=(
  srun
  --partition="$PARTITION"
  --nodes="$NODE_COUNT"
  --ntasks-per-node=1
  --gpus=0
  --mem=4G
  --cpus-per-task=2
  --time=00:30:00
  --job-name=stage-fastafd-source
  "${node_args[@]}"
  bash -c
  'set -euxo pipefail
   SOURCE_TREE=$1
   CODE_DIR=$2
   if [[ -e "$CODE_DIR" && ! -d "$CODE_DIR/.git" ]]; then
     echo "Refusing to replace non-Git path: $CODE_DIR" >&2
     exit 1
   fi
   if [[ ! -d "$CODE_DIR/.git" ]]; then
     mkdir -p "$(dirname "$CODE_DIR")"
     rsync -a --exclude=/.conda-env/ --exclude=/cache/ --exclude=/models/ \
       --exclude=/results/ "$SOURCE_TREE/" "$CODE_DIR/"
   fi
   if [[ ! -x "$CODE_DIR/run_col_rocm.sh" ]]; then
     echo "Existing checkout predates run_col_rocm.sh: $CODE_DIR" >&2
     echo "Update it explicitly; setup will not overwrite it." >&2
     exit 1
   fi
   hostname
   git -C "$CODE_DIR" status --short --branch'
  _ "$SOURCE_TREE" "$CODE_DIR"
)
show_command "${source_cmd[@]}"
"${source_cmd[@]}"

echo
echo "Validating the saved environment in a short one-GPU job."
VALIDATION_IMAGE=$LOCAL_IMAGE
VALIDATION_CODE=$CODE_DIR
validate_cmd=(
  srun
  --partition="$PARTITION"
  --nodes=1
  --gpus=1
  --mem=220G
  --cpus-per-task=20
  --time=00:20:00
  --job-name=validate-fastafd
  "${validation_node_args[@]}"
  --container-image="$VALIDATION_IMAGE"
  --container-workdir="$VALIDATION_CODE"
  bash -c
  'set -euxo pipefail
   ENV_PREFIX=$1
   CODE_DIR=$2
   if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
     export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
   fi
   unset ROCR_VISIBLE_DEVICES
   export PYTHONPATH="$CODE_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
   "$ENV_PREFIX/bin/python" "$CODE_DIR/scripts/check_rocm_runtime.py"
   "$ENV_PREFIX/bin/python" -c "import minisgl, torch; print(torch.cuda.get_device_name(0))"'
  _ "$ENV_PREFIX" "$VALIDATION_CODE"
)
show_command "${validate_cmd[@]}"
"${validate_cmd[@]}"

echo
echo "Checking for a complete $MODEL_REPO snapshot on $TARGET_NODE."
model_check_cmd=(
  srun
  --partition="$PARTITION"
  --nodes=1
  --ntasks=1
  --gpus=0
  --mem=2G
  --cpus-per-task=1
  --time=00:10:00
  --job-name=check-gpt-oss-120b
  "${node_args[@]}"
  python3 -c
  'import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
required = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
missing = [name for name in required if not (d / name).is_file()]
weights = []
if not missing:
    try:
        weights = sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values()))
    except Exception as exc:
        print(f"Invalid model index: {exc}", file=sys.stderr)
        raise SystemExit(1)
    missing += [name for name in weights if not (d / name).is_file() or (d / name).stat().st_size == 0]
if missing or not weights:
    print("Incomplete model snapshot; missing: " + ", ".join(missing[:20]), file=sys.stderr)
    raise SystemExit(1)
print(f"Complete model snapshot: {len(weights)} indexed weight shards in {d}")'
  "$MODEL_DIR"
)
show_command "${model_check_cmd[@]}"

if "${model_check_cmd[@]}"; then
  echo "Reusing the complete existing model; no Hugging Face write is needed."
  srun --nodes=1 --ntasks=1 --gpus=0 --mem=1G --cpus-per-task=1 \
    "${node_args[@]}" du -sh "$MODEL_DIR"
else
  echo "Model is absent or incomplete; downloading/resuming it now."
model_cmd=(
  srun
  --partition="$PARTITION"
  --nodes="$NODE_COUNT"
  --ntasks-per-node=1
  --gpus=0
  --mem=16G
  --cpus-per-task=8
  --time=08:00:00
  --job-name=stage-gpt-oss-120b
  "${node_args[@]}"
  --container-image="$LOCAL_IMAGE"
  bash -c
  'set -euxo pipefail
   MODEL_REPO=$1
   MODEL_DIR=$2
   ENV_PREFIX=$3
   mkdir -p "$MODEL_DIR"
   "$ENV_PREFIX/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_DIR"
   hostname
   du -sh "$MODEL_DIR"'
  _ "$MODEL_REPO" "$MODEL_DIR" "$ENV_PREFIX"
)
show_command "${model_cmd[@]}"
"${model_cmd[@]}"
fi

echo
echo "Setup complete. Start an interactive container with:"
echo "  $TOOLS_DIR/shell.sh 4"
