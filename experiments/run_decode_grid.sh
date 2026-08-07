#!/usr/bin/env bash
# Reproduce the decode grid used in the gpt-oss MI300X performance log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.conda-env}"
MODEL="${MODEL:-}"
PORT="${PORT:-19295}"
CONFIG="${CONFIG:-colocated_tp4_packed}"
ISLS="${ISLS:-2048 8192 32768}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32 64}"
OSL="${OSL:-32}"
OUT_DIR="${OUT_DIR:-$ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)-$CONFIG}"

[[ -x "$ENV_PREFIX/bin/python" ]] || { echo "missing environment: $ENV_PREFIX" >&2; exit 1; }
[[ -n "$MODEL" && -e "$MODEL" ]] || { echo "MODEL must point to model weights" >&2; exit 1; }
mkdir -p "$OUT_DIR"

for isl in $ISLS; do
  for concurrency in $CONCURRENCIES; do
    out="$OUT_DIR/isl${isl}_c${concurrency}.json"
    echo "[grid] isl=$isl concurrency=$concurrency out=$out"
    "$ENV_PREFIX/bin/python" "$ROOT/experiments/serve_bench.py" \
      --port "$PORT" --isl "$isl" --osl "$OSL" \
      --concurrency "$concurrency" --config "$CONFIG" --out "$out"
  done
done

echo "[grid] complete: $OUT_DIR"
