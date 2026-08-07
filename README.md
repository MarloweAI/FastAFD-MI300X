# FastAFD for AMD MI300X

Minimal, reproducible ROCm/gfx942 runtime extracted from the
[`amd-mi300x`](https://github.com/hao-ai-lab/FastAFD) port of FastAFD. The
upstream project targets NVIDIA Blackwell; this repository contains the AMD
runtime needed for colocated serving and the experimental single-node AFD path.

The serving source is unchanged from AMD port commit `9512cb8`. This repository
does **not** contain model weights, generated caches, development logs, reports,
profiler output, Git history from the development branch, or CUDA-only DeepEP
and DeepGEMM sources.

## Validated configuration

- Linux x86-64, AMD MI300X (`gfx942`), 192 GiB HBM per GPU.
- System ROCm `7.2.4` in `/opt/rocm`.
- Python `3.12.13`.
- PyTorch `2.10.0+rocm7.0` (reports HIP `7.0.51831`).
- `pytorch-triton-rocm 3.5.1`, Ray `2.54.0`, transformers `4.57.3`,
  tvm-ffi `0.1.9`.

Host tools: Miniforge/Conda, `git`, `curl`, `bc`, `iproute2`, `rocminfo`, and
`rocm-smi`. The user must have permission to access `/dev/kfd` and `/dev/dri`.

## 1. Clone and create the environment

```bash
git clone https://github.com/MarloweAI/FastAFD-MI300X.git
cd FastAFD-MI300X
./bootstrap_rocm.sh
export ENV_PREFIX="$PWD/.conda-env"
```

The bootstrap script creates `.conda-env`, installs the local `minisgl` package,
and verifies ROCm, dependencies, and `gfx942`. Set `CONDA_EXE` if `conda` is not
on `PATH`, `ROCM_PATH` if ROCm is not in `/opt/rocm`, or `ENV_PREFIX` to use a
different environment location.

`environment.rocm7.yml` is the validated environment definition inherited from
the port. `environment.rocm7.pinned.yml` pins the direct packages to the exact
versions observed on the source MI300X machine.

## 2. Download gpt-oss-120b weights

The model is approximately 61 GiB. Keep at least 70 GiB free for the snapshot
and additional space for the environment and JIT cache.

```bash
mkdir -p models
"$ENV_PREFIX/bin/hf" download openai/gpt-oss-120b \
  --local-dir "$PWD/models/gpt-oss-120b"
export MODEL="$PWD/models/gpt-oss-120b"
```

If Hugging Face requests authentication:

```bash
"$ENV_PREFIX/bin/hf" auth login
```

The launch scripts have development-machine defaults, so always export both
`ENV_PREFIX` and `MODEL` on another machine.

## 3. Run the recommended colocated server

Packed MXFP4 is the mature gpt-oss path. It keeps the expert weights packed and
has been correctness/performance validated at TP2 and TP4.

```bash
export ENV_PREFIX="$PWD/.conda-env"
export MODEL="$PWD/models/gpt-oss-120b"
export MINISGL_MXFP4_PACKED=1
TP=4 GPUS=0,1,2,3 GRAPH_MAX_BS=32 PORT=19295 ./run_col_rocm.sh
```

First launch JIT-compiles HIP/Triton kernels into `cache/`; allow roughly 1–3
minutes beyond model loading. Later launches reuse the cache. With packed MXFP4
and the default memory ratio, keep `GRAPH_MAX_BS<=96` until the destination has
been benchmarked. For high concurrency, reduce KV allocation with, for example,
`EXTRA_ARGS="--memory-ratio 0.5"`.

Smoke-test from another shell:

```bash
cd FastAFD-MI300X
./ask_rocm.sh "What is the capital of France?"
```

## 4. Reproduce the measured experiments

All experiment commands assume the colocated server is already running and
`ENV_PREFIX` and `MODEL` are exported.

Single serving point (TTFT, ITL/TPOT, and output throughput):

```bash
mkdir -p results
"$ENV_PREFIX/bin/python" experiments/serve_bench.py \
  --port 19295 --isl 8192 --osl 32 --concurrency 32 \
  --config colocated_tp4_packed --out results/tp4_isl8192_c32.json
```

The original benchmark implementation is preserved byte-for-byte and carries a
stale Qwen label in its JSON `model` field. For these commands, `config` and the
exported `MODEL` path identify the actual gpt-oss run; do not use that legacy
label when aggregating results.

Decode grid used by the performance investigation:

```bash
PORT=19295 CONFIG=colocated_tp4_packed \
ISLS="2048 8192 32768" CONCURRENCIES="1 8 32 64" OSL=32 \
./experiments/run_decode_grid.sh
```

Unique-prompt prefill TTFT at 2K, 8K, and 32K:

```bash
PORT=19295 "$ENV_PREFIX/bin/python" experiments/prefill_ttft.py
```

Greedy token-ID alignment against Hugging Face is much more memory-intensive:
the server and reference must use disjoint visible GPUs. For example, run a TP2
server on GPUs 0–1, then use GPUs 2–3 for the BF16 HF reference:

```bash
HIP_VISIBLE_DEVICES=2,3 HF_DEVICE=auto PORT=19295 MODEL="$MODEL" \
  "$ENV_PREFIX/bin/python" experiments/t10_hf_alignment.py --n 32
```

The checked-in 32-prompt/32-token reference avoids loading HF alongside the
server:

```bash
PORT=19295 MODEL="$MODEL" "$ENV_PREFIX/bin/python" \
  experiments/t10_hf_alignment.py --n 32 --max-tokens 32 \
  --hf-in experiments/refs/gptoss120b_hf_ref_n32_t32.json
```

## 5. Experimental AMD AFD

The AMD AFD implementation replaces NVIDIA-only DeepEP/DeepGEMM with RCCL
transport and Triton experts. It is functionally validated, but it runs eager,
defaults to one microbatch, and is substantially slower than colocated TP.
Packed MXFP4 is implemented on this path but has not been exercised live, so do
not enable `MINISGL_MXFP4_PACKED` for the AFD experiment.

```bash
unset MINISGL_MXFP4_PACKED
export ENV_PREFIX="$PWD/.conda-env"
export MODEL="$PWD/models/gpt-oss-120b"
ATTN_TP=2 MLP_TP=2 MLP_EP=2 GPUS=0,1,2,3 \
MEM_RATIO=0.5 NUM_MB=1 GRAPH_MAX_BS=0 PORT=19297 \
./run_afd_rocm.sh
```

For a 1-attention + 3-FFN layout, use `ATTN_TP=1 MLP_TP=3 MLP_EP=3`.
`MLP_EP=1` in that layout silently produces incorrect results.

## Important ROCm rules

- Do not install NVIDIA NCCL packages, CUDA `triton`, FlashInfer, DeepEP, or
  DeepGEMM in this environment.
- Set only `HIP_VISIBLE_DEVICES` through the `GPUS` launcher option. Do not also
  set `ROCR_VISIBLE_DEVICES`; the filters compose and can hide every GPU.
- Colocated TP>1 uses the port's pynccl/RCCL wrapper because torch distributed's
  watchdog breaks HIP graph capture.
- Do not copy `cache/` unless source revision, Python, torch, Triton, ROCm, and
  GPU architecture all match. Recompiling is the portable choice.
- AFD currently does not recover from a client disconnect; restart it after an
  aborted request.

See [AGENTS.md](AGENTS.md) for a concise coding-agent recreation checklist.
