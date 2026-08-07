#!/usr/bin/env bash
# Send a prompt to a running minisgl server (see run_rocm.sh) and print the reply.
#
#   ./ask_rocm.sh "What is the capital of France?"
#   MAX_TOKENS=128 ./ask_rocm.sh "Explain MoE routing in two sentences."
#   PORT=19300 TEMP=0.7 ./ask_rocm.sh "Write a haiku about GPUs."
set -euo pipefail

PORT="${PORT:-19295}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMP="${TEMP:-0}"
PROMPT="${*:-What is the capital of France? Answer with just the city name.}"

# Build the request with python's json so quotes/newlines in the prompt are safe.
BODY=$(PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" TEMP="$TEMP" python3 -c '
import json, os
print(json.dumps({
    "model": "q",
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMP"]),
}))')

START=$(date +%s.%N)
RESP=$(curl -sS --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY")
END=$(date +%s.%N)

# The endpoint streams SSE ("data: {json}") chunks; concatenate the content deltas.
# Each chunk is parsed as JSON, NOT regexed: a naive
# `grep -o '"content": "[^"]*"'` truncates at the first ESCAPED quote, so a model
# emitting the token `"expert"` silently came out as `xpert\`.
PARSED=$(printf '%s' "$RESP" | python3 -c '
import sys, json
pieces = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        continue
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        continue
    for choice in chunk.get("choices") or []:
        piece = (choice.get("delta") or {}).get("content")
        if piece:
            pieces.append(piece)
print(len(pieces))                 # line 1: number of content chunks
sys.stdout.write("".join(pieces))  # remainder: the text itself
')
NCHUNK=${PARSED%%$'\n'*}
TEXT=${PARSED#*$'\n'}

if [[ -z "$TEXT" ]]; then
  echo "no content returned; raw response:" >&2
  printf '%s\n' "$RESP" | head -20 >&2
  exit 1
fi

printf '%s\n' "$TEXT"

# Wall-clock includes prefill, HTTP and detokenization, so this UNDERSTATES decode
# rate — and the very first request after startup also pays Triton JIT for the MoE
# kernel at that shape. Use it as a smoke signal, not a benchmark
# (dev_log/qwen/wave64_fix.md sec 6).
printf '\n[%s chunks, %.2fs wall, ~%.1f tok/s incl. prefill+HTTP]\n' \
  "$NCHUNK" "$(echo "$END - $START" | bc)" \
  "$(echo "scale=2; $NCHUNK / ($END - $START)" | bc)" >&2
