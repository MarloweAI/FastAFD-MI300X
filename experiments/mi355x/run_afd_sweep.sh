#!/usr/bin/env bash
set -euo pipefail

readonly inferencex_commit=770268c51c2b368e9d669096041c003520f14c3a
readonly aiperf_commit=062a5de92c8ac8a0a6dd5d2a7fb9a539a147f3d9
readonly model_revision=b5c939de8f754692c1647ca79fbf85e8c1e70f8a

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
inferencex=${INFERENCEX_CHECKOUT:-/workspace/$USER/inferencex-770268c}
image=${INFERENCEX_IMAGE:-/workspace/images/vllm_vllm-openai-rocm_v0.22.0_c3f18c9b.sqsh}
hf_cache=${HF_HUB_CACHE_HOST:-/workspace/hf_cache}
python_overlay=${FASTAFD_PYTHON_OVERLAY:-/workspace/$USER/FastAFD-MI355X-python-v1}
partition=${SLURM_PARTITION:-mi355x}
node=${SLURM_NODELIST:-}
srun_node_args=()
[[ -z $node ]] || srun_node_args+=(--nodelist="$node")
run_label=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
result_root=${RESULT_ROOT:-/workspace/$USER/FastAFD-MI355X-results/$run_label}
concurrencies=${CONCURRENCIES:-"4 8 16 32 64 128"}
isl=${ISL:-8192}
osl=${OSL:-1024}
prompt_multiplier=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}
cpus_per_task=${SLURM_CPUS_PER_TASK:-124}

default_splits=(7:1 6:2 5:3 4:4 3:5 2:6 1:7)
if (( $# > 0 )); then
  splits=("$@")
else
  splits=("${default_splits[@]}")
fi

[[ -r $image ]] || { echo "missing image: $image" >&2; exit 1; }
[[ -d $python_overlay/ray ]] || {
  echo "missing AFD Python overlay; run experiments/mi355x/bootstrap_overlay.sh" >&2; exit 1;
}
[[ -d $inferencex/.git ]] || { echo "missing InferenceX checkout: $inferencex" >&2; exit 1; }
[[ $(git -C "$inferencex" rev-parse HEAD) == "$inferencex_commit" ]] || {
  echo "InferenceX checkout is not pinned to $inferencex_commit" >&2; exit 1;
}
[[ $(git -C "$inferencex/utils/aiperf" rev-parse HEAD) == "$aiperf_commit" ]] || {
  echo "aiperf submodule is not pinned to $aiperf_commit" >&2; exit 1;
}
[[ $(<"$hf_cache/models--openai--gpt-oss-120b/refs/main") == "$model_revision" ]] || {
  echo "GPT-OSS model cache is not pinned to $model_revision" >&2; exit 1;
}
[[ -z $node || $node == marlowe-mi355x-* ]] || {
  echo "SLURM_NODELIST must name the physical MI355X comparison node" >&2; exit 1;
}
mkdir -p "$result_root"

for split in "${splits[@]}"; do
  [[ $split =~ ^([1-7]):([1-7])$ ]] || { echo "invalid split: $split" >&2; exit 2; }
  attention=${BASH_REMATCH[1]}
  experts=${BASH_REMATCH[2]}
  (( attention + experts == 8 )) || { echo "split must sum to 8: $split" >&2; exit 2; }
  point_dir="$result_root/afd-${attention}a-${experts}f"
  mkdir -p "$point_dir"

  printf 'running AFD %s on %s; output=%s\n' "$split" "${node:-any available node}" "$point_dir"
  # The inner script is single-quoted so Slurm-exported values expand in-container.
  # shellcheck disable=SC2016
  srun \
    --partition="$partition" \
    "${srun_node_args[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:8 \
    --cpus-per-task="$cpus_per_task" \
    --exclusive \
    --time="${SLURM_TIME:-04:00:00}" \
    --output="$point_dir/slurm-%j.out" \
    --container-image="$image" \
    --container-mounts="$repo:/fastafd,$inferencex:/inferencex,$hf_cache:/mnt/hf_hub_cache,$python_overlay/ray:/usr/local/lib/python3.12/dist-packages/ray,$point_dir:/results" \
    --container-mount-home \
    --container-writable \
    --container-workdir=/fastafd \
    --container-remap-root \
    --no-container-entrypoint \
    --export="ALL,ATTN_GPUS=$attention,FFN_GPUS=$experts,CONCURRENCIES=$concurrencies,ISL=$isl,OSL=$osl,BENCH_NUM_PROMPTS_MULTIPLIER=$prompt_multiplier,PORT=8888,HF_HUB_CACHE=/mnt/hf_hub_cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1,PYTHONDONTWRITEBYTECODE=1" \
    bash -lc '
      set -euo pipefail
      model=/mnt/hf_hub_cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
      [[ -r $model/config.json ]] || { echo "missing pinned model snapshot: $model"; exit 1; }
      export MODEL=$model
      export EVAL_ONLY=false
      export MODEL_PREFIX=gptoss FRAMEWORK=fastafd PRECISION=fp4
      export SPEC_DECODING=none DISAGG=false RUNNER_TYPE=mi355x IMAGE=fastafd-mi355x
      export TP=8 EP_SIZE=$FFN_GPUS DP_ATTENTION=true
      export MINISGL_M2N_FIXED_SHAPE=${MINISGL_M2N_FIXED_SHAPE:-1}
      export GRAPH_MAX_BS=${GRAPH_MAX_BS:-128}
      export NUM_MB=${NUM_MB:-2}

      cleanup() {
        if [[ -n ${server_pid:-} ]]; then kill "$server_pid" 2>/dev/null || true; fi
        ray stop --force >/dev/null 2>&1 || true
      }
      trap cleanup EXIT INT TERM

      /fastafd/run_afd_mi355x.sh > /results/server.log 2>&1 &
      server_pid=$!
      ready=0
      for _ in $(seq 1 900); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$server_pid" 2>/dev/null || { tail -n 200 /results/server.log; exit 1; }
        sleep 2
      done
      (( ready == 1 )) || { echo "server readiness timeout"; tail -n 200 /results/server.log; exit 1; }

      python3 /fastafd/scripts/check_mi355x_runtime.py > /results/runtime.json
      curl -fsS "http://127.0.0.1:$PORT/v1/models" > /results/models.json
      set +e
      PORT=$PORT MODEL=$model python3 /fastafd/experiments/t10_hf_alignment.py \
        --n 4 --max-tokens 32 \
        --hf-in /fastafd/experiments/refs/gptoss120b_hf_ref_n32_t32.json \
        2>&1 | tee /results/token-alignment.txt
      alignment_status=${PIPESTATUS[0]}
      set -e
      printf "%s\n" "$alignment_status" > /results/token-alignment.exitcode
      if (( alignment_status != 0 )) && [[ ${ALLOW_ALIGNMENT_FAILURE:-0} != 1 ]]; then
        echo "token alignment failed; refusing to benchmark" >&2
        exit "$alignment_status"
      fi
      if (( alignment_status != 0 )); then
        echo "WARNING: benchmarking despite failed token alignment because ALLOW_ALIGNMENT_FAILURE=1" \
          | tee /results/PROVISIONAL_RESULTS.txt
      fi

      source /inferencex/benchmarks/benchmark_lib.sh
      for conc in $CONCURRENCIES; do
        name="afd_gptoss120b_${ISL}i_${OSL}o_${ATTN_GPUS}a_${FFN_GPUS}f_c${conc}"
        mkdir -p "/results/c${conc}"
        # With the validated eight-request internal admission cap, high external
        # concurrencies intentionally queue and can take over an hour at c128.
        benchmark_timeout_seconds=${BENCHMARK_TIMEOUT_SECONDS:-7200}
        run_benchmark_serving \
          --model "$model" --port "$PORT" --backend vllm \
          --input-len "$ISL" --output-len "$OSL" --random-range-ratio 0.8 \
          --num-prompts "$((conc * BENCH_NUM_PROMPTS_MULTIPLIER))" --max-concurrency "$conc" \
          --result-filename "$name" --result-dir "/results/c${conc}" \
          --bench-serving-dir /inferencex --server-pid "$server_pid" &
        benchmark_pid=$!
        benchmark_started=$SECONDS
        while kill -0 "$benchmark_pid" 2>/dev/null; do
          if (( SECONDS - benchmark_started >= benchmark_timeout_seconds )); then
            printf "benchmark exceeded %ss at concurrency %s\n" \
              "$benchmark_timeout_seconds" "$conc" \
              | tee "/results/c${conc}/TIMEOUT.txt"
            kill -TERM "$benchmark_pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$benchmark_pid" 2>/dev/null || true
            wait "$benchmark_pid" 2>/dev/null || true
            exit 124
          fi
          sleep 5
        done
        wait "$benchmark_pid"
      done
    '
done

printf 'AFD sweep complete: %s\n' "$result_root"
