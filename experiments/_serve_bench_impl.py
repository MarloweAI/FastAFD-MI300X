#!/usr/bin/env python
"""Serving benchmark in the schema of dev_log/qwen/qwen3_mi300x_realchip_handoff.md sec 6.

Uses `/generate` (raw prompt, no chat template) with `ignore_eos` so ISL and OSL are
controlled exactly -- the chat endpoint injects template tokens and stops early on EOS,
neither of which is acceptable when the measured points must line up 1:1 with simulated
ones.

Reports TTFT (time to first token), ITL/TPOT (inter-token latency during decode) and
aggregate output throughput, per that document's definitions.

  python dev_log/probes/serve_bench.py --port 19600 --isl 8192 --osl 256 \
      --concurrency 16 --config afd_1a3f --out result.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request

MODEL_DIR = "/home/marlowe/models/Qwen3-30B-A3B-Instruct-2507"


def build_prompt(isl: int) -> tuple[str, int]:
    """A prompt that tokenizes to (as close as possible to) `isl` tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Distinct-ish filler so the radix cache cannot collapse it; AFD forces
    # cache_type=naive anyway, but the colocated baseline must not get a free ride.
    words = [f"item{i} " for i in range(isl * 2)]
    text = "".join(words)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < isl:
        raise RuntimeError(f"filler too short: {len(ids)} < {isl}")
    text = tok.decode(ids[:isl])
    actual = len(tok(text, add_special_tokens=False)["input_ids"])
    return text, actual


def one_request(port: int, prompt: str, osl: int) -> tuple[float, list[float], int]:
    """Returns (ttft_s, inter_token_latencies_s, n_tokens)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps({"prompt": prompt, "max_tokens": osl, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    stamps: list[float] = []
    with urllib.request.urlopen(req, timeout=7200) as resp:
        # NOTE: /generate streams the RAW incremental text (`yield f"data: {...}"` in
        # api_server.py:177), not JSON like /v1/chat/completions does. So each
        # `data:` line is one token event and must NOT be json.loads()'d.
        for raw in resp:
            line = raw.decode(errors="replace")
            if not line.startswith("data:"):
                continue                      # continuation of a token containing '\n'
            if line[5:].strip() == "[DONE]":
                break
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            stamps.append(now)
    if ttft is None:
        raise RuntimeError("no tokens returned")
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return ttft, itls, len(stamps)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k] * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--config", required=True, help='e.g. "afd_1a3f" / "colocated_tp4"')
    ap.add_argument("--hardware", default="4xMI300X")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    prompt, actual_isl = build_prompt(args.isl)
    print(f"prompt: requested isl={args.isl} actual={actual_isl} tokens", flush=True)

    if args.warmup:
        # Warm up AT THE TARGET CONCURRENCY, not with a single request.
        #
        # This used to be `one_request(port, prompt, min(8, osl))`, which only warms the
        # bs=1 shapes. Everything first encountered at concurrency C -- Triton autotune
        # for the batched MoE/prefill shapes, graph-bucket selection, allocator growth --
        # was then charged to the first measured request, and because a concurrent batch
        # completes together it landed on ALL of them.
        #
        # It produced a 12x phantom: gpt-oss at c32 measured ttft p50 = 952-1058 ms on the
        # first pass and 76 ms on an immediate rerun of the identical config against the
        # same server. The tight p50/p95 spread (952.5 / 958.4) was the tell -- a real
        # scheduling problem spreads TTFT across the batch, a one-off cost shared by one
        # batch does not. See dev_log/gpt_oss_120b/00_README.md §15.
        C_warm = args.concurrency
        with cf.ThreadPoolExecutor(C_warm) as ex:
            list(
                ex.map(
                    lambda _: one_request(args.port, prompt, min(8, args.osl)),
                    range(C_warm),
                )
            )

    C = args.concurrency
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(C) as ex:
        futs = [ex.submit(one_request, args.port, prompt, args.osl) for _ in range(C)]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results]
    all_itls = [x for r in results for x in r[1]]
    total_tokens = sum(r[2] for r in results)
    # TPOT per request = (e2e - ttft) / (n-1), i.e. mean decode step time
    tpots = [
        (sum(r[1]) / len(r[1])) for r in results if r[1]
    ]

    out = {
        "run_id": args.run_id or f"{args.config}_isl{args.isl}_c{C}",
        "config": args.config,
        "model": "Qwen/Qwen3-30B-A3B (local Instruct-2507)",
        "precision": {"weights": "bf16", "kv_cache": "bf16"},
        "hardware": args.hardware,
        "stack": {"name": "minisgl-FastAFD-rocm", "version": "amd-mi300x", "rocm": "7.2.4"},
        "isl": actual_isl,
        "isl_requested": args.isl,
        "osl": args.osl,
        "max_concurrency": C,
        "num_prompts": C,
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) * 1000,
            "p50": pct(ttfts, 50),
            "p95": pct(ttfts, 95),
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpots) * 1000 if tpots else None,
            "p50": pct(tpots, 50),
            "p95": pct(tpots, 95),
        },
        "itl_ms": {
            "mean": statistics.fmean(all_itls) * 1000 if all_itls else None,
            "p50": pct(all_itls, 50),
        },
        "output_throughput_tok_s": total_tokens / wall,
        "request_throughput_req_s": C / wall,
        "wall_s": wall,
        "total_output_tokens": total_tokens,
        "notes": "",
    }
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
