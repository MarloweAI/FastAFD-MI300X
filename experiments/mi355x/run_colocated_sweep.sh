#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
inferencex=${INFERENCEX_CHECKOUT:-/workspace/$USER/inferencex-770268c}
image=${INFERENCEX_IMAGE:-/workspace/images/vllm_vllm-openai-rocm_v0.22.0_c3f18c9b.sqsh}
hf_cache=${HF_HUB_CACHE_HOST:-/workspace/hf_cache}
node=${SLURM_NODELIST:-}
srun_node_args=()
[[ -z $node ]] || srun_node_args+=(--nodelist="$node")
run_label=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)-colocated}
result_root=${RESULT_ROOT:-/workspace/$USER/FastAFD-MI355X-results/$run_label/colocated-tp8}
concurrencies=${CONCURRENCIES:-"4 8"}
isl=${ISL:-8192}
osl=${OSL:-1024}
prompt_multiplier=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}
cpus_per_task=${SLURM_CPUS_PER_TASK:-124}

mkdir -p "$result_root"
srun \
  --partition="${SLURM_PARTITION:-mi355x}" "${srun_node_args[@]}" \
  --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task="$cpus_per_task" --exclusive \
  --time="${SLURM_TIME:-02:00:00}" --output="$result_root/slurm-%j.out" \
  --container-image="$image" \
  --container-mounts="$repo:/fastafd,$inferencex:/inferencex,$hf_cache:/mnt/hf_hub_cache,$result_root:/results" \
  --container-mount-home --container-writable --container-workdir=/fastafd \
  --container-remap-root --no-container-entrypoint \
  --export="ALL,ISL=$isl,OSL=$osl,CONCURRENCIES=$concurrencies,BENCH_NUM_PROMPTS_MULTIPLIER=$prompt_multiplier,PORT=8888,HF_HUB_CACHE=/mnt/hf_hub_cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONDONTWRITEBYTECODE=1" \
  bash -lc '
    set -euo pipefail
    model=/mnt/hf_hub_cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
    cleanup() { if [[ -n ${server_pid:-} ]]; then kill "$server_pid" 2>/dev/null || true; fi; }
    trap cleanup EXIT INT TERM
    export ENV_PREFIX=/usr PYTHON_BIN=/usr/bin/python3 MODEL=$model TP=8 GRAPH_MAX_BS=128
    export MINISGL_MXFP4_PACKED=1
    export EXTRA_ARGS="--max-seq-len-override $((ISL + OSL + 256)) --memory-ratio 0.82"
    /fastafd/run_col_rocm.sh > /results/server.log 2>&1 & server_pid=$!
    for _ in $(seq 1 900); do
      curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
      kill -0 "$server_pid" 2>/dev/null || { tail -200 /results/server.log; exit 1; }
      sleep 2
    done
    source /inferencex/benchmarks/benchmark_lib.sh
    for conc in $CONCURRENCIES; do
      dir=/results/c$conc; mkdir -p "$dir"
      run_benchmark_serving --model "$model" --port "$PORT" --backend vllm \
        --input-len "$ISL" --output-len "$OSL" --random-range-ratio 0.8 \
        --num-prompts "$((conc * BENCH_NUM_PROMPTS_MULTIPLIER))" --max-concurrency "$conc" \
        --result-filename "colocated_gptoss120b_${ISL}i_${OSL}o_tp8_c${conc}" \
        --result-dir "$dir" --bench-serving-dir /inferencex --server-pid "$server_pid"
    done
  '

printf 'colocated sweep complete: %s\n' "$result_root"
