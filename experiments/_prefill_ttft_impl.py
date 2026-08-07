#!/usr/bin/env python
"""Prefill TTFT with UNIQUE prompts, so the radix cache cannot inflate the result.

The grid in 16_projection_vs_silicon.md measured TTFT non-monotonic in ISL (32,768 faster
than 2,048). `serve_bench.build_prompt` emits `item0 item1 ...` filler, so each longer ISL is
a strict prefix-extension of the shorter one and reuses its KV. This probe gives every
(ISL, repeat) its own random token content, and reports pass 1 vs pass 2 separately so
first-encounter compilation is visible rather than blended in.
"""
from __future__ import annotations
import argparse, json, random, time, urllib.request
from transformers import AutoTokenizer

MODEL = "/home/marlowe/models/gpt-oss-120b"

def ttft(port, prompt):
    body = {"model": "g", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0.0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            s = line.decode().strip()
            if s.startswith("data: ") and s[6:] != "[DONE]":
                if json.loads(s[6:])["choices"][0].get("delta", {}).get("content"):
                    return (time.perf_counter() - t0) * 1000
    raise RuntimeError("no content")

def build_exact(tok, rng, target: int, tries: int = 40) -> tuple[str, int]:
    """Text whose SERVER-SIDE token count is exactly `target`, template included.

    Two effects used to make the real prefill longer than the nominal ISL, and comparing that
    against a projection point for the nominal length overstated our time:

      * random token ids do not round-trip decode->encode, inflating the count ~3%;
      * the harmony chat template adds 67 tokens the earlier probe never counted.

    At nominal 8192 the server actually prefilled 8508. So build against what
    TokenizeManager will do -- apply_chat_template then encode (tokenizer/tokenize.py:19-28) --
    and iterate on the content length until the total lands exactly on target.
    """
    def server_len(text: str) -> int:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return len(tok.encode(rendered))

    k = target                      # content ids to draw; converges in a few steps
    ids = [rng.randrange(1000, tok.vocab_size - 1000) for _ in range(k)]
    for _ in range(tries):
        text = tok.decode(ids)
        n = server_len(text)
        if n == target:
            return text, n
        delta = n - target
        if delta > 0:
            ids = ids[:-delta] if delta < len(ids) else ids[: len(ids) // 2]
        else:
            ids = ids + [rng.randrange(1000, tok.vocab_size - 1000) for _ in range(-delta)]
    # Fell short of exact: report what we got rather than silently comparing the wrong length.
    text = tok.decode(ids)
    return text, server_len(text)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=19311)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(1234)
    print(f"{'ISL':>7}{'pass1':>10}{'pass2':>10}{'pass3':>10}{'server_tok':>12}"
          f"   (ms, unique content each pass, exact server-side length)")
    for isl in (2048, 8192, 32768):
        row = []
        for p in range(3):
            # Fresh content every pass -> no prefix any earlier prompt can supply.
            text, n = build_exact(tok, rng, isl)
            row.append((ttft(a.port, text), n))
        got = {n for _, n in row}
        flag = "" if got == {isl} else f"  <- NOT EXACT: {sorted(got)}"
        print(f"{isl:>7}" + "".join(f"{t:>10.1f}" for t, _ in row)
              + f"{row[-1][1]:>12}{flag}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
