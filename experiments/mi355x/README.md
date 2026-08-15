# GPT-OSS-120B FastAFD Pareto on Colovore MI355X

This experiment compares FastAFD with the original InferenceX/vLLM Pareto on
the same physical 8×MI355X node and the same random 8192-input/1024-output
workload. It is a CDNA4 experiment, not an MI300X behavior-preservation port.

## Experiment contract

- A split `A:B` means `A` TP1 attention workers and `B` TP1 FFN workers.
- Every run consumes all eight physical GPUs. The `B` FFN workers form one
  full-world expert-parallel group for GPT-OSS's 128 experts.
- Uneven partitions use `ceil(128 / B)` local slots. The last shard is padded;
  only the real expert range appears in the routing map.
- Splits: `7:1`, `6:2`, `5:3`, `4:4`, `3:5`, `2:6`, `1:7`.
- Concurrency: `4`, `8`, `16`, `32`, `64`, `128` for every split (42 points).
- Client: pinned InferenceX `benchmark_serving.py`, random dataset, seed 0,
  range ratio 0.8, request rate infinity, ignore EOS, `10 × concurrency`
  measured requests, and `2 × concurrency` warmups. These are the values in
  the pinned executable recipe.
- Pareto x-axis: total input+output token throughput divided by all 8 physical
  GPUs. Pareto y-axis: `1000 / median_tpot_ms`, in output tokens/s/user.
- Correctness gate: API/model smoke plus greedy token-ID alignment against the
  checked-in HF reference before any timing point for each split.

## Pinned inputs

| Component | Value |
|---|---|
| InferenceX | `770268c51c2b368e9d669096041c003520f14c3a` |
| aiperf submodule | `062a5de92c8ac8a0a6dd5d2a7fb9a539a147f3d9` |
| vLLM image | `/workspace/images/vllm_vllm-openai-rocm_v0.22.0_c3f18c9b.sqsh` |
| Image digest | `sha256:c3f18c9baf778cb4f9456a0f161e658fb45d70e3cd534dc3d1c55fac478d03bd` |
| GPT-OSS-120B revision | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| AITER in validated image | `amd-aiter==0.1.13` |

The runtime checker records the actual source, host, Python, PyTorch, ROCm,
AITER, and GPU architecture beside every split. AITER 0.1.13 is used because it
is baked into the exact reproduced-vLLM image and already contains the gfx950
A16W4 shuffles and fused MoE API this implementation calls. A newer AITER may
be evaluated as a separate, explicitly labeled environment; never replace it
silently inside an existing comparison.

## Run

Choose one physical node for both systems. The default is the node from which
the command is submitted; override it explicitly when needed:

```bash
export SLURM_NODELIST=marlowe-mi355x-4
export INFERENCEX_CHECKOUT=/workspace/$USER/inferencex-770268c
export HARDWARE_ACCESS_CHECKOUT=/workspace/$USER/hardware-access
export RUN_LABEL=$(date -u +%Y%m%dT%H%M%SZ)
```

Build the pinned AFD-only Ray overlay once. Ray is installed with `--no-deps`
so it reuses, rather than overrides, the packages pinned in the immutable vLLM
comparison image. The sweep mounts only the `ray` package directory at the
image's normal site-packages location so Ray's subprocess workers resolve the
same package without a broad `PYTHONPATH` overlay:

```bash
experiments/mi355x/bootstrap_overlay.sh
```

Reproduce all nine original vLLM points on that node:

```bash
experiments/mi355x/run_vllm_baseline.sh
```

Run one AFD split first. This performs environment validation, model/API smoke,
token alignment, then all six concurrency points while keeping the server hot:

```bash
experiments/mi355x/run_afd_sweep.sh 4:4
```

Run all 42 AFD points:

```bash
experiments/mi355x/run_afd_sweep.sh
```

Launchers leave node selection to Slurm by default. Set `SLURM_NODELIST` only
for a deliberately pinned same-node experiment; otherwise the first eligible
MI355X node is used.

The launcher defaults to packed MXFP4, AITER, fixed-shape RCCL M:N transport,
two microbatches, decode graph buckets through 128, and eight internally active
requests. InferenceX concurrency is not reduced: requests above eight queue at
the server. The cap is required because continuous admission above eight can
wedge the current collective schedule after a partially drained batch. Override
tuning knobs only under a new run label, for example:

```bash
RUN_LABEL=aiter-eager-check GRAPH_MAX_BS=0 NUM_MB=1 \
  MINISGL_M2N_FIXED_SHAPE=0 experiments/mi355x/run_afd_sweep.sh 4:4
```

## Generate the comparison

After reviewing the raw JSON and token-alignment logs, pass the same-node vLLM
review CSV and the AFD run root to the plotter:

```bash
python3 experiments/mi355x/plot_pareto.py \
  --afd-root "/workspace/$USER/FastAFD-MI355X-results/$RUN_LABEL" \
  --vllm-csv "/workspace/$USER/hardware-access/sites/colovore/health-tests/inference/reports/gptoss120b-8k1k.csv" \
  --output-csv results/mi355x/pareto.csv \
  --output-svg results/mi355x/pareto.svg
```

The SVG shows published vLLM, reproduced same-node vLLM, every AFD point,
per-split AFD frontiers, and the combined AFD envelope. Dominated AFD points are
retained with low opacity. The normalized CSV records source paths and frontier
membership, so the picture is auditable.

## Acceptance and failure policy

A split is not benchmarkable until its runtime manifest says all visible GPUs
are `gfx950`, the AITER symbols are present, and token alignment passes. Server
death, readiness timeout, missing raw JSON, NaN/zero TPOT, or any fallback from
packed MXFP4 is a failed point, not a number to omit. Retain `server.log`, the
Slurm log, `runtime.json`, `models.json`, `token-alignment.txt`, and every raw
client result under the run label.
