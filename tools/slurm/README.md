# FastAFD on the MI300X Slurm cluster

This directory contains a deliberately small and transparent workflow:

- `setup.sh` builds and stages the environment once.
- `shell.sh` reserves GPUs and opens an interactive container shell.
- `env.sh` sets the four paths FastAFD needs inside the container.

The scripts print the underlying Slurm commands. There is no SSH daemon inside
the container and no hidden Git synchronization. You SSH to the head node, then
use `srun --pty` (through `shell.sh`) to obtain a shell in the GPU container.

`experiments/profile_report.py` is a tracked analysis helper available in every
FastAFD checkout. It can show aggregate kernel costs, the
timestamp-ordered execution stream, one or several inferred TP ranks, and the
repeated sliding-window/full-attention layer pattern:

```bash
experiments/profile_report.py results/profiles/RUN
experiments/profile_report.py results/profiles/RUN \
  --view timeline --rank 0,1 --limit 100
experiments/profile_report.py results/profiles/RUN \
  --view pattern --rank all
experiments/profile_report.py results/profiles/RUN \
  --view pattern --rank 0 --step 0
```

`--step N` prints every kernel in one complete decode iteration and annotates
the prologue, epilogue, layer number, and sliding-window/full-attention type.
Add `--markers-only` for the older compact one-attention-kernel-per-layer view,
or `--kernel REGEX` to filter the complete iteration without changing its
inferred boundaries.

## Storage layout

| Content | Location |
|---|---|
| Canonical, immutable container | `/nfs/containers/sqsh/fastafd-rocm724-v1.sqsh` |
| Runnable per-node container copy | `/scratch/images/fastafd-rocm724-v1.sqsh` |
| Active Git checkout | `/scratch/$USER/FastAFD-MI300X` |
| Model | `/scratch/models/gpt-oss-120b` |
| JIT cache | `/scratch/$USER/FastAFD-MI300X/cache` |
| Results and profiles | `/scratch/$USER/FastAFD-MI300X/results` |

`/scratch` is node-local and is not backed up. Commit and push source changes.
Copy only results that need to survive into your NFS home.

## 1. One-time setup

Review the variables at the beginning of `setup.sh`, especially the image name.
The source used for initial staging is the current repository checkout. Keep
that management checkout on NFS so it is visible from the head and GPU nodes,
for example:

```text
/nfs/home/$USER/FastAFD-MI300X
```

The management checkout already points at the MarloweAI repository. Verify the
remote and your authenticated account before setup; the node-local copies inherit
this configuration:

```bash
cd /nfs/home/$USER/FastAFD-MI300X
git remote -v
gh auth status
```

Only change `origin` if you intentionally want to work through a separate fork.

Then run:

```bash
cd /nfs/home/$USER/FastAFD-MI300X
./tools/slurm/setup.sh
```

`setup.sh` asks Slurm for one GPU first and keeps that allocation for the whole
setup, so another job cannot take the node between phases. Slurm chooses an
available node. To request a particular node instead:

```bash
FASTAFD_NODE=mi300x-01 ./tools/slurm/setup.sh
```

This is the slow path. The outer allocation reserves one GPU, 20 CPUs, and 220
GiB of RAM for up to 12 hours. Individual setup phases use only what they need,
but the GPU stays reserved until image staging, source staging, live validation,
and model download/resume all finish on that same node. Override the limit with
`FASTAFD_SETUP_TIME=HH:MM:SS` (the cluster maximum is 12 hours).

To create a new image instead of reusing `v1`, choose a new name explicitly:

```bash
FASTAFD_IMAGE_NAME=fastafd-rocm724-v2.sqsh ./tools/slurm/setup.sh
```

Select that staged version in later development shells:

```bash
FASTAFD_IMAGE=/scratch/images/fastafd-rocm724-v2.sqsh ./tools/slurm/shell.sh 4
```

## 2. Frequent development path

From the head node, request four GPUs for the default six hours:

```bash
cd /nfs/home/$USER/FastAFD-MI300X
FASTAFD_NODE=mi300x-02 ./tools/slurm/shell.sh 4
```

For an explicit `FASTAFD_NODE`, `shell.sh` first checks that node's local
source and approximately 61 GiB runtime model. If either is absent, it exits and
prints the node-specific `setup.sh` command. If only the image is absent or has
the wrong size, it atomically stages the immutable image from NFS before
invoking Pyxis. Run setup once per node, and avoid omitting `FASTAFD_NODE` unless
all eligible GPU nodes are prepared.

`shell.sh` does not download models. The only model download is performed by
`setup.sh`, whose defaults are `FASTAFD_MODEL_REPO=openai/gpt-oss-120b` and
`FASTAFD_MODEL=/scratch/models/gpt-oss-120b`. The shell checks
`FASTAFD_MODEL`, and `env.sh` exposes the same path as `MODEL` to FastAFD.

For a different model, use the same node-local destination during setup and
every subsequent shell:

```bash
FASTAFD_NODE=mi300x-02 \
FASTAFD_MODEL_REPO=ORG/MODEL \
FASTAFD_MODEL=/scratch/models/my-model \
./tools/slurm/setup.sh

FASTAFD_NODE=mi300x-02 \
FASTAFD_MODEL=/scratch/models/my-model \
./tools/slurm/shell.sh 4
```

Other examples:

```bash
./tools/slurm/shell.sh 1
./tools/slurm/shell.sh 8 08:00:00
```

The script prints and executes a command equivalent to:

```bash
srun \
  --nodes=1 \
  --gpus=4 \
  --time=06:00:00 \
  --job-name=fastafd-dev \
  --container-image=/scratch/images/fastafd-rocm724-v1.sqsh \
  --container-workdir=/scratch/$USER/FastAFD-MI300X \
  --pty bash -i
```

Once the container prompt appears:

The startup message identifies the FastAFD container immediately, and the prompt
is shortened to `(fastafd:mi300x-01)`. The environment remains explicit rather
than being loaded automatically. Interactive shells also define `h=history` and
use confirmation aliases for `cp`, `mv`, and `rm` (`-i`); scripts are unaffected.

```bash
source tools/slurm/env.sh
git status
python scripts/check_rocm_runtime.py
python -c 'import torch; print(torch.cuda.device_count())'
```

Exit the shell to release the Slurm allocation:

```bash
exit
```

## 3. Start the colocated server

`run_col_rocm.sh` starts the colocated tensor-parallel inference server. With a
four-GPU allocation:

```bash
export MINISGL_MXFP4_PACKED=1

TP=4 \
GRAPH_MAX_BS=32 \
PORT=19295 \
./run_col_rocm.sh
```

The `TP` value must be no larger than the GPU count requested from `shell.sh`.
For example, `shell.sh 2` supports `TP=1` or `TP=2`, while `TP=4` requires
`shell.sh 4`. The launcher checks the visible GPU count and exits before spawning
workers when the allocation is too small.

Do not set `GPUS`, `HIP_VISIBLE_DEVICES`, or `ROCR_VISIBLE_DEVICES`; Slurm has
already selected the allocated GPUs. `env.sh` and the launchers translate
Slurm's `ROCR_VISIBLE_DEVICES` value to `HIP_VISIBLE_DEVICES`, which is the name
accepted by both Ray and torch.

The first launch loads the model and JIT-compiles kernels. It remains in the
foreground until interrupted with Ctrl-C.

### Smoke test and serving benchmark

Run the server in the background when testing it from the same shell:

```bash
mkdir -p results

TP=4 GRAPH_MAX_BS=32 PORT=19295 ./run_col_rocm.sh \
  >results/colocated-server.log 2>&1 &
server_pid=$!

curl -fsS http://127.0.0.1:19295/health
curl -fsS http://127.0.0.1:19295/v1/models | python -m json.tool
PORT=19295 ./ask_rocm.sh "What is the capital of France?"
```

The health endpoint returns `{"status":"ok"}` only after the backend is ready.
If startup fails, inspect `tail -n 100 results/colocated-server.log`.

The serving benchmark sends concurrent HTTP requests to the already-running
server. It does not start a server. It uses the tokenizer at the exported
`MODEL` path, so source `env.sh` (or export `MODEL` explicitly) in the benchmark
shell:

```bash
"$ENV_PREFIX/bin/python" experiments/serve_bench.py \
  --port 19295 \
  --isl 8192 \
  --osl 32 \
  --concurrency 32 \
  --config colocated_tp4_packed \
  --out results/tp4_isl8192_c32.json
```

It reports time to first token, inter-token latency/time per output token, and
aggregate output/request throughput. `--config` is only a label; server settings
come from `run_col_rocm.sh` and its environment variables.

Stop the background server:

```bash
kill "$server_pid"
wait "$server_pid"
```

This explicit stop is preferred because it reports the server's final status in
the current shell. As a safety net, container shells opened by `shell.sh` also
install an EXIT handler: leaving the shell sends `SIGTERM` to its background
jobs, waits up to 30 seconds for graceful shutdown and log flushing, and only
then force-stops jobs that have not exited. Slurm releases the allocation after
the container shell and its processes finish.

For profiling, stay in the same interactive container so the code, model, and
JIT cache remain on local NVMe. Inspect the installed profiler options first,
then put the explicit server command after `--`:

```bash
rocprofv3 --help
rocprofv3 -- bash -lc \
  'TP=4 GRAPH_MAX_BS=32 PORT=19295 ./run_col_rocm.sh'
```

Keep profiler output under `results/` (or another `/scratch` path), not NFS.

## 4. Start the experimental AFD server

`run_afd_rocm.sh` separates attention and FFN work. For a four-GPU allocation:

```bash
unset MINISGL_MXFP4_PACKED

ATTN_TP=2 \
MLP_TP=2 \
MLP_EP=2 \
MEM_RATIO=0.5 \
NUM_MB=1 \
GRAPH_MAX_BS=0 \
PORT=19297 \
./run_afd_rocm.sh
```

Test it with:

```bash
PORT=19297 ./ask_rocm.sh "What is the capital of France?"
```

## Run directly as part of the container invocation

For a foreground colocated server without first opening an interactive shell:

```bash
srun \
  --nodes=1 \
  --gpus=4 \
  --time=06:00:00 \
  --container-image=/scratch/images/fastafd-rocm724-v1.sqsh \
  --container-workdir=/scratch/$USER/FastAFD-MI300X \
  bash -lc '
    source tools/slurm/env.sh
    export MINISGL_MXFP4_PACKED=1
    TP=4 GRAPH_MAX_BS=32 PORT=19295 ./run_col_rocm.sh
  '
```

For AFD:

```bash
srun \
  --nodes=1 \
  --gpus=4 \
  --time=06:00:00 \
  --container-image=/scratch/images/fastafd-rocm724-v1.sqsh \
  --container-workdir=/scratch/$USER/FastAFD-MI300X \
  bash -lc '
    source tools/slurm/env.sh
    unset MINISGL_MXFP4_PACKED
    ATTN_TP=2 MLP_TP=2 MLP_EP=2 \
      MEM_RATIO=0.5 NUM_MB=1 GRAPH_MAX_BS=0 PORT=19297 \
      ./run_afd_rocm.sh
  '
```

## Git and durable results

Source transfer between the two local checkouts is explicit:

```bash
git status
git pull --ff-only
git add -p
git commit
git push
```

The setup and shell scripts never reset, clean, stash, pull, or overwrite an
existing checkout.

Copy selected results to NFS when they need to survive:

```bash
mkdir -p /nfs/home/$USER/fastafd-results
cp results/tp4_isl8192_c32.json /nfs/home/$USER/fastafd-results/
```

## Debugging

Useful commands inside the container:

```bash
hostname
env | rg 'SLURM|ROCR|HIP'
amd-smi monitor
rocminfo | rg gfx942
git status
ls -lh "$MODEL"
du -sh cache results
```

Check jobs from the head node:

```bash
squeue --me
sacct -j JOB_ID
scancel JOB_ID
```
