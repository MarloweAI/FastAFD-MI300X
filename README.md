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

## Slurm container workflow

The tracked helpers in [`tools/slurm/`](tools/slurm/README.md) provide the
recommended workflow on the Marlowe MI300X cluster. Keep the management clone
on NFS, while active code, models, caches, and profiles remain on node-local
scratch:

| Content | Location |
|---|---|
| Management Git checkout | `/nfs/home/$USER/FastAFD-MI300X` |
| Immutable shared image | `/nfs/containers/sqsh/fastafd-rocm724-v1.sqsh` |
| Staged runnable image | `/scratch/images/fastafd-rocm724-v1.sqsh` |
| Active checkout | `/scratch/$USER/FastAFD-MI300X` |
| Model | `/scratch/models/gpt-oss-120b` |
| JIT cache and results | active checkout `cache/` and `results/` |

One-time setup from the head node builds the shared image, then stages the
image, source, and approximately 61 GiB runtime model on one selected node. Run
it once for each node whose local scratch you intend to use:

```bash
git clone https://github.com/MarloweAI/FastAFD-MI300X.git \
  /nfs/home/$USER/FastAFD-MI300X
cd /nfs/home/$USER/FastAFD-MI300X
FASTAFD_NODE=mi300x-01 ./tools/slurm/setup.sh
FASTAFD_NODE=mi300x-02 ./tools/slurm/setup.sh
```

The frequent development path reserves GPUs and opens the already-staged
container. Slurm chooses a node unless `FASTAFD_NODE` is set:

```bash
cd /nfs/home/$USER/FastAFD-MI300X
FASTAFD_NODE=mi300x-01 ./tools/slurm/shell.sh 4
```

Because `/scratch` is node-local, `shell.sh` checks an explicit node before
launch. If its source or model is absent, it exits with the exact setup command.
If only the immutable image is absent, it stages that single file atomically
from NFS. Avoid leaving `FASTAFD_NODE` unset unless every eligible GPU node has
already been prepared.

`shell.sh` never downloads model weights. Only `setup.sh` invokes Hugging Face:
by default it downloads `openai/gpt-oss-120b` into
`/scratch/models/gpt-oss-120b`. The shell preflight checks that directory, and
`env.sh` exports it as `MODEL` for the launchers and benchmarks. To use another
model, keep the repository and local directory explicit and consistent:

```bash
FASTAFD_NODE=mi300x-02 \
FASTAFD_MODEL_REPO=ORG/MODEL \
FASTAFD_MODEL=/scratch/models/my-model \
./tools/slurm/setup.sh

FASTAFD_NODE=mi300x-02 \
FASTAFD_MODEL=/scratch/models/my-model \
./tools/slurm/shell.sh 4
```

After `source tools/slurm/env.sh`, the second command automatically exposes
`MODEL=/scratch/models/my-model` inside the container.

Inside the container, explicitly load the environment and work from scratch:

```bash
source tools/slurm/env.sh
git status
python scripts/check_rocm_runtime.py
```

Exit the shell to gracefully stop its background jobs and release the Slurm
allocation. The setup scripts never reset, pull, or overwrite an existing Git
checkout. `/scratch` is node-local and not backed up, so commit source and copy
selected results to NFS. Use a new `FASTAFD_IMAGE_NAME` when rebuilding an
immutable image; see the detailed tools README for image-version examples.

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

`TP` cannot exceed the GPUs visible to the process. The launcher checks
`torch.cuda.device_count()` before spawning workers and exits with a clear error
if, for example, `TP=4` is used inside a two-GPU Slurm allocation.

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

Start a fresh colocated server under `rocprofv3`, warm the exact request shape,
run one measured benchmark, and stop the server that the script started:

```bash
TP=4 GRAPH_MAX_BS=32 OSL=256 CONCURRENCY=32 \
./experiments/profile_steady_rocm.sh results/profiles/tp4-c32
```

The script refuses to run if any MiniSGL server is already visible in the
current container/allocation, or if `PORT` is occupied. It prints the existing
PID and exits without killing anything. A server isolated inside another Slurm
job is outside both the process namespace and allocated GPU set. Otherwise the
script records kernel dispatches, RCCL calls, memory copies, aggregate
statistics, raw CSV, and Perfetto output. It always gracefully stops only the
profiler/server process group that it created.

Raw traces cover startup, warmup, and measurement. The script records the exact
post-warmup measurement window and creates `measurement_kernel_times.csv` plus
`measurement_kernel_stats.csv`, containing kernel names and runtimes for that
window only. TP runs also create `measurement_rccl_times.csv` and
`measurement_rccl_stats.csv` with RCCL function-call timings. RCCL GPU kernels
such as `rcclGenericKernel` remain in the kernel tables as well. `OSL=256` is the
default; increase it for a longer steady decode. Override the workload with
`PORT`, `ISL`, `OSL`, or `CONCURRENCY`. Use a new or empty output directory each
time.

Use the CSV-aware report tool instead of `column -s,`, because demangled C++
kernel names contain commas. Aggregate kernel cost, chronological launch order,
and repeated decode-layer structure are available directly from the profile
directory:

```bash
run=results/profiles/tp4-c32

./experiments/profile_report.py "$run" --view summary --rank 0,1
./experiments/profile_report.py "$run" --view timeline --rank 0,1 --limit 100
./experiments/profile_report.py "$run" --view pattern --rank all
./experiments/profile_report.py "$run" --view pattern --rank 0 --step 0
```

`summary` aggregates names and durations and can sort by `total`, `average`,
`max`, `calls`, or `name`. `timeline` orders individual launches by GPU start
timestamp; ranks and queues may overlap. `pattern` reads `config.json`, detects
the repeated sliding-window/full-attention layer order, and validates complete
decode iterations against attention markers. `--step N` prints every kernel in
one iteration, annotated as prologue, layer/type, or epilogue. Add
`--markers-only` for one attention marker per layer, or `--kernel REGEX` to
filter the detailed step without changing its inferred boundaries.

The benchmark loads its tokenizer from the exported `MODEL` path and records
that path in the JSON `model` field. It exits clearly if `MODEL` is absent or is
not a local directory. Set `MINISGL_MXFP4_PACKED=1` in the benchmark shell when
that is how the server was launched so the JSON precision label also matches.

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
- Match the launcher layout to the allocation. Colocated mode requires at least
  `TP` visible GPUs; AFD requires at least `ATTN_TP + MLP_TP`. Both launchers
  validate this before starting worker processes.
- Colocated TP>1 uses the port's pynccl/RCCL wrapper because torch distributed's
  watchdog breaks HIP graph capture.
- Do not copy `cache/` unless source revision, Python, torch, Triton, ROCm, and
  GPU architecture all match. Recompiling is the portable choice.
- AFD currently does not recover from a client disconnect; restart it after an
  aborted request.

See [AGENTS.md](AGENTS.md) for a concise coding-agent recreation checklist.
