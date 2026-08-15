#!/usr/bin/env bash
# Build the small AFD-only dependency overlay beside the immutable vLLM image.
set -euo pipefail

image=${INFERENCEX_IMAGE:-/workspace/images/vllm_vllm-openai-rocm_v0.22.0_c3f18c9b.sqsh}
overlay=${FASTAFD_PYTHON_OVERLAY:-/workspace/$USER/FastAFD-MI355X-python-v1}
partition=${SLURM_PARTITION:-mi355x}
node=${SLURM_NODELIST:-$(hostname -s)}
mkdir -p "$overlay"

srun \
  --partition="$partition" --nodelist="$node" \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=00:20:00 \
  --job-name=fastafd-overlay \
  --container-image="$image" \
  --container-mounts="$overlay:/overlay" \
  --container-writable --container-remap-root --no-container-entrypoint \
  bash -lc 'python3 -m pip install --target=/overlay --upgrade --no-deps "ray==2.54.0"'

printf 'AFD Python overlay ready: %s\n' "$overlay"
