#!/usr/bin/env bash
# Start a colocated MiniSGL server under rocprofv3, warm it, measure it, and stop it.
#
#   TP=4 GRAPH_MAX_BS=32 ./experiments/profile_steady_rocm.sh
#   OSL=512 CONCURRENCY=32 ./experiments/profile_steady_rocm.sh results/profiles/my-run
#
# Optional environment: ENV_PREFIX, MODEL, PORT, TP, GRAPH_MAX_BS, ISL, OSL,
# CONCURRENCY, CONFIG, HARDWARE, SERVER_START_TIMEOUT, SHUTDOWN_TIMEOUT.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.conda-env}"
MODEL="${MODEL:-}"
PORT="${PORT:-19295}"
TP="${TP:-}"
GRAPH_MAX_BS="${GRAPH_MAX_BS:-32}"
ISL="${ISL:-8192}"
OSL="${OSL:-256}"
CONCURRENCY="${CONCURRENCY:-32}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-90}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="${1:-${OUT_DIR:-$ROOT/results/profiles/steady-$STAMP}}"

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

for command_name in curl pgrep ps rocprofv3 setsid ss; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "ERROR: missing command: $command_name" >&2
    exit 1
  }
done

# Refuse before creating output or starting anything. This deliberately checks
# every MiniSGL server owned by this user and visible in the current container,
# not only PORT, so it cannot disturb a server in this allocation.
mapfile -t existing_servers < <(
  pgrep -u "$(id -u)" -af '(^|/)(python|python3)([.][0-9]+)? -m minisgl([[:space:]]|$)' || true
)
if (( ${#existing_servers[@]} > 0 )); then
  echo "ERROR: a MiniSGL server is already running in this container/allocation:" >&2
  printf '  %s\n' "${existing_servers[@]}" >&2
  echo "Nothing was started or stopped. Exit that server yourself, then rerun this script." >&2
  exit 3
fi
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "ERROR: TCP port $PORT is already in use in this container/allocation." >&2
  echo "Nothing was started or stopped. Choose another PORT or stop its owner yourself." >&2
  exit 3
fi

for name in PORT GRAPH_MAX_BS ISL OSL CONCURRENCY SERVER_START_TIMEOUT SHUTDOWN_TIMEOUT; do
  value=${!name}
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer; got $value" >&2
    exit 2
  fi
done
if (( PORT < 1 || PORT > 65535 || ISL < 1 || OSL < 1 || CONCURRENCY < 1 )); then
  echo "ERROR: PORT must be 1-65535 and ISL, OSL, and CONCURRENCY must be positive." >&2
  exit 2
fi
if (( SERVER_START_TIMEOUT < 1 || SHUTDOWN_TIMEOUT < 1 )); then
  echo "ERROR: SERVER_START_TIMEOUT and SHUTDOWN_TIMEOUT must be positive." >&2
  exit 2
fi

[[ -x "$ENV_PREFIX/bin/python" ]] || {
  echo "ERROR: missing Python environment: $ENV_PREFIX" >&2
  exit 1
}
[[ -n "$MODEL" && -d "$MODEL" ]] || {
  echo "ERROR: MODEL must point to a local model directory; got: ${MODEL:-<unset>}" >&2
  exit 1
}

ALLOCATED_GPUS=$(
  "$ENV_PREFIX/bin/python" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null
) || {
  echo "ERROR: could not determine the GPUs visible in this container." >&2
  exit 1
}
if ! [[ "$ALLOCATED_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: no GPUs are visible in this container." >&2
  exit 1
fi
TP="${TP:-$ALLOCATED_GPUS}"
if ! [[ "$TP" =~ ^[1-9][0-9]*$ ]] || (( TP > ALLOCATED_GPUS )); then
  echo "ERROR: TP must be between 1 and the $ALLOCATED_GPUS visible GPUs; got TP=$TP." >&2
  exit 2
fi

# rocprofiler-sdk in ROCm 7.2.4 needs libdw. New FastAFD images install it
# system-wide. This fallback keeps already-staged v1 images usable.
if ! ldconfig -p 2>/dev/null | grep -q 'libdw[.]so[.]1'; then
  ROCprof_FALLBACK="${ROCPROF_LIBDW_DIR:-/scratch/$USER/rocprof-runtime/lib}"
  if [[ -e "$ROCprof_FALLBACK/libdw.so.1" ]]; then
    export LD_LIBRARY_PATH="$ROCprof_FALLBACK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using rocprof libdw fallback: $ROCprof_FALLBACK"
  else
    echo "ERROR: rocprofv3 requires libdw.so.1, but this container does not have it." >&2
    echo "Rebuild the FastAFD image, or set ROCPROF_LIBDW_DIR to a directory containing libdw.so.1." >&2
    exit 1
  fi
fi

if [[ -d "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: output directory is not empty: $OUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR/rocprof"

DEFAULT_CONFIG="colocated_tp$TP"
[[ -n "${MINISGL_MXFP4_PACKED:-}" ]] && DEFAULT_CONFIG+="_packed"
CONFIG="${CONFIG:-${DEFAULT_CONFIG}_profiled}"
HARDWARE="${HARDWARE:-${TP}xMI300X}"

PROFILE_PID=""
SERVER_OWNED=0

stop_owned_server() {
  local waited=0
  local wait_status=0
  (( SERVER_OWNED == 1 )) || return 0
  SERVER_OWNED=0

  # Signal the MiniSGL parent first so its own handler can stop and reap workers.
  # A process group signal would interrupt every multiprocessing child at once.
  if process_is_running "$PROFILE_PID"; then
    echo "Stopping the profiled server gracefully (PID $PROFILE_PID)."
    kill -INT "$PROFILE_PID" 2>/dev/null || true
    while process_group_is_running "$PROFILE_PID" && (( waited < SHUTDOWN_TIMEOUT )); do
      sleep 1
      ((waited += 1))
    done
  fi
  if process_group_is_running "$PROFILE_PID"; then
    echo "WARNING: graceful shutdown exceeded ${SHUTDOWN_TIMEOUT}s; sending SIGTERM." >&2
    kill -TERM -- "-$PROFILE_PID" 2>/dev/null || true
    waited=0
    while process_group_is_running "$PROFILE_PID" && (( waited < 10 )); do
      sleep 1
      ((waited += 1))
    done
  fi
  if process_group_is_running "$PROFILE_PID"; then
    echo "WARNING: shutdown still incomplete; force-stopping the owned process group." >&2
    kill -KILL -- "-$PROFILE_PID" 2>/dev/null || true
  fi
  set +e
  wait "$PROFILE_PID" 2>/dev/null
  wait_status=$?
  set -e
  printf '%s\n' "$wait_status" >"$OUT_DIR/server_exit_status.txt"
  # ROCm 7.2.4 can make uvicorn return 1 while restoring its signal handler,
  # after it has already logged a completed application shutdown and flushed
  # profiler output. Preserve the status for diagnosis without calling that a
  # failed shutdown.
  if (( wait_status != 0 && wait_status != 130 && wait_status != 143 )) && \
      ! grep -q 'Application shutdown complete' "$OUT_DIR/server.log"; then
    echo "WARNING: profiled server/profiler exited with status $wait_status." >&2
  elif (( wait_status != 0 && wait_status != 130 && wait_status != 143 )); then
    echo "Server shutdown completed; profiler wrapper status was $wait_status (recorded)."
  fi
}

process_is_running() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null | awk 'NR == 1 {print $1}')
  [[ -n "$state" && "$state" != Z* ]]
}

process_group_is_running() {
  ps -eo pgid=,stat= | awk -v group="$1" \
    '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  stop_owned_server
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT HUP
trap 'exit 143' TERM

echo "FastAFD steady-state ROCm profile"
echo "  server   : colocated TP=$TP, port $PORT, graphs=$GRAPH_MAX_BS"
echo "  workload : ISL=$ISL OSL=$OSL concurrency=$CONCURRENCY"
echo "  output   : $OUT_DIR"
echo "[1/5] Starting a new server under rocprofv3."

# setsid gives the profiler and every server child a private process group. The
# cleanup handler signals only this owned group; it never searches for or kills
# unrelated processes.
export ENV_PREFIX MODEL PORT TP GRAPH_MAX_BS
setsid rocprofv3 \
  --disable-signal-handlers true \
  --kernel-trace \
  --rccl-trace \
  --memory-copy-trace \
  --stats \
  --summary-per-domain \
  --summary-units usec \
  --output-format csv pftrace \
  --output-file 'server-%pid%' \
  --output-directory "$OUT_DIR/rocprof" \
  -- "$ROOT/run_col_rocm.sh" \
  >"$OUT_DIR/server.log" 2>&1 &
PROFILE_PID=$!
SERVER_OWNED=1
printf '%s\n' \
  "profile_pid=$PROFILE_PID" "visible_gpus=$ALLOCATED_GPUS" "tp=$TP" \
  "port=$PORT" "graph_max_bs=$GRAPH_MAX_BS" "isl=$ISL" "osl=$OSL" \
  "concurrency=$CONCURRENCY" "config=$CONFIG" "model=$MODEL" \
  >"$OUT_DIR/metadata.txt"

echo "[2/5] Waiting for /health (timeout ${SERVER_START_TIMEOUT}s)."
deadline=$((SECONDS + SERVER_START_TIMEOUT))
while ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 -- "-$PROFILE_PID" 2>/dev/null; then
    echo "ERROR: server exited before becoming ready. Last log lines:" >&2
    tail -n 100 "$OUT_DIR/server.log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "ERROR: server was not ready after ${SERVER_START_TIMEOUT}s. Last log lines:" >&2
    tail -n 100 "$OUT_DIR/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "[3/5] Warming the exact target shape."
"$ENV_PREFIX/bin/python" "$ROOT/experiments/serve_bench.py" \
  --port "$PORT" --isl "$ISL" --osl "$OSL" \
  --concurrency "$CONCURRENCY" --warmup 1 \
  --config "${CONFIG}_warmup" --hardware "$HARDWARE" \
  --out "$OUT_DIR/warmup.json"

echo "[4/5] Running the measured request."
MEASURE_START_NS=$("$ENV_PREFIX/bin/python" -c 'import time; print(time.monotonic_ns())')
"$ENV_PREFIX/bin/python" "$ROOT/experiments/serve_bench.py" \
  --port "$PORT" --isl "$ISL" --osl "$OSL" \
  --concurrency "$CONCURRENCY" --warmup 0 \
  --config "$CONFIG" --hardware "$HARDWARE" \
  --out "$OUT_DIR/benchmark.json"
MEASURE_END_NS=$("$ENV_PREFIX/bin/python" -c 'import time; print(time.monotonic_ns())')
printf 'measurement_start_monotonic_ns=%s\nmeasurement_end_monotonic_ns=%s\n' \
  "$MEASURE_START_NS" "$MEASURE_END_NS" >"$OUT_DIR/measurement_window.txt"

echo "[5/5] Stopping the owned server and finalizing profiler output."
stop_owned_server

# Produce a compact, measurement-only table. Raw rocprof CSV and Perfetto files
# remain available for startup/warmup inspection. rocprof timestamps and
# time.monotonic_ns() use the same monotonic clock on this host.
"$ENV_PREFIX/bin/python" - "$OUT_DIR" "$MEASURE_START_NS" "$MEASURE_END_NS" "$TP" <<'PY'
import csv
import pathlib
import sys
from collections import defaultdict

out = pathlib.Path(sys.argv[1])
window_start, window_end = map(int, sys.argv[2:4])
tp = int(sys.argv[4])
rows = []
for path in out.joinpath("rocprof").rglob("*kernel_trace.csv"):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if start >= window_start and end <= window_end:
                rows.append(
                    {
                        "Source": str(path.relative_to(out)),
                        "Agent_Id": row["Agent_Id"],
                        "Queue_Id": row["Queue_Id"],
                        "Kernel_Name": row["Kernel_Name"],
                        "Start_Timestamp": start,
                        "End_Timestamp": end,
                        "DurationNs": end - start,
                        "DurationUs": f"{(end - start) / 1000:.3f}",
                    }
                )

if not rows:
    raise SystemExit(
        "ERROR: profiler produced no kernel timing rows inside the measured window"
    )

fields = list(rows[0])
with out.joinpath("measurement_kernel_times.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["Start_Timestamp"]))

grouped = defaultdict(list)
for row in rows:
    grouped[(row["Agent_Id"], row["Kernel_Name"])].append(row["DurationNs"])
total = sum(row["DurationNs"] for row in rows)
summary = []
for (agent, name), durations in grouped.items():
    subtotal = sum(durations)
    summary.append(
        {
            "Agent_Id": agent,
            "Kernel_Name": name,
            "Calls": len(durations),
            "TotalDurationNs": subtotal,
            "AverageNs": f"{subtotal / len(durations):.3f}",
            "MinNs": min(durations),
            "MaxNs": max(durations),
            "Percent": f"{100 * subtotal / total:.4f}",
        }
    )
summary.sort(key=lambda row: int(row["TotalDurationNs"]), reverse=True)
with out.joinpath("measurement_kernel_stats.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(f"Verified {len(rows)} measured kernel dispatches with names and runtimes.")
print(f"Top kernel: {summary[0]['Kernel_Name']}")
print(f"Top kernel total: {int(summary[0]['TotalDurationNs']) / 1e6:.3f} ms")

rccl_rows = []
for path in out.joinpath("rocprof").rglob("*rccl_api_trace.csv"):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if start >= window_start and end <= window_end:
                rccl_rows.append(
                    {
                        "Source": str(path.relative_to(out)),
                        "Process_Id": row["Process_Id"],
                        "Thread_Id": row["Thread_Id"],
                        "Function": row["Function"],
                        "Start_Timestamp": start,
                        "End_Timestamp": end,
                        "DurationNs": end - start,
                        "DurationUs": f"{(end - start) / 1000:.3f}",
                    }
                )

if tp > 1 and not rccl_rows:
    raise SystemExit("ERROR: TP > 1 but no RCCL timing rows were captured")
if rccl_rows:
    with out.joinpath("measurement_rccl_times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rccl_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rccl_rows, key=lambda row: row["Start_Timestamp"]))

    rccl_grouped = defaultdict(list)
    for row in rccl_rows:
        rccl_grouped[row["Function"]].append(row["DurationNs"])
    rccl_summary = []
    rccl_total = sum(row["DurationNs"] for row in rccl_rows)
    for function, durations in rccl_grouped.items():
        subtotal = sum(durations)
        rccl_summary.append(
            {
                "Function": function,
                "Calls": len(durations),
                "TotalDurationNs": subtotal,
                "AverageNs": f"{subtotal / len(durations):.3f}",
                "MinNs": min(durations),
                "MaxNs": max(durations),
                "Percent": f"{100 * subtotal / rccl_total:.4f}",
            }
        )
    rccl_summary.sort(key=lambda row: int(row["TotalDurationNs"]), reverse=True)
    with out.joinpath("measurement_rccl_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rccl_summary[0]))
        writer.writeheader()
        writer.writerows(rccl_summary)
    print(f"Verified {len(rccl_rows)} measured RCCL API calls with runtimes.")
PY

trap - EXIT INT TERM HUP
echo "Profile complete: $OUT_DIR"
echo "  measured kernels : $OUT_DIR/measurement_kernel_times.csv"
echo "  measured summary : $OUT_DIR/measurement_kernel_stats.csv"
if [[ -s "$OUT_DIR/measurement_rccl_times.csv" ]]; then
  echo "  measured RCCL    : $OUT_DIR/measurement_rccl_times.csv"
  echo "  RCCL summary     : $OUT_DIR/measurement_rccl_stats.csv"
fi
echo "  raw trace         : $OUT_DIR/rocprof/"
echo "  Perfetto          : find $OUT_DIR/rocprof -name '*.pftrace'"
