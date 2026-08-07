#!/usr/bin/env python
"""T-10: greedy token-ID alignment between minisgl and HF `transformers`.

The primary correctness gate. Upstream's harness (scripts/validate/*) scores against
a vLLM baseline configured with deepep_low_latency + deep_gemm, neither of which
exists on ROCm, so the reference is HF instead.

Method
------
* Both sides see byte-identical input ids: the prompt is rendered once with
  `tokenizer.apply_chat_template(...)`, and that exact string is what minisgl is
  asked to continue and what HF is fed.
* Greedy on both sides (`temperature=0`, `do_sample=False`).
* Compare **token ids**, not text -- detokenization can hide or invent differences.
* On mismatch, report the first divergent position and HF's top-5 logit gap there,
  so a near-tie can be told apart from a real bug.

Usage
-----
  # HF reference on a different card from the server
  HF_DEVICE=cuda:1 PORT=19310 python dev_log/probes/t10_hf_alignment.py --n 32

  # A model too large for one card: HF_DEVICE is passed straight to `device_map`, so
  # "auto" shards it over whatever HIP_VISIBLE_DEVICES exposes. gpt-oss-120b in bf16 is
  # ~234 GB and needs two MI300X.
  HIP_VISIBLE_DEVICES=1,2 HF_DEVICE=auto PORT=19310 \
    MODEL=/home/marlowe/models/gpt-oss-120b \
    python dev_log/probes/t10_hf_alignment.py --n 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import torch

MODEL = os.environ.get(
    "MODEL", "/home/marlowe/models/Qwen3-30B-A3B-Instruct-2507"
)
PORT = int(os.environ.get("PORT", "19310"))
HF_DEVICE = os.environ.get("HF_DEVICE", "cuda:1")

# Deliberately varied: factual, arithmetic, code, list, reasoning, long-ish context.
# Short prompts alone would under-test the attention mask.
PROMPTS = [
    "What is the capital of France? Answer with just the city name.",
    "List the first 10 prime numbers, comma separated.",
    "What is 17 * 23? Show only the number.",
    "In one sentence, why is the sky blue?",
    "Write a Python function that reverses a string.",
    "Name three primary colors.",
    "What is the chemical symbol for gold?",
    "Translate 'good morning' into Spanish.",
    "What year did the Apollo 11 mission land on the moon?",
    "Explain what a mixture-of-experts model is, in two sentences.",
    "Sort these numbers ascending: 42, 7, 19, 3, 88.",
    "What is the derivative of x^3 with respect to x?",
    "Give the first 8 Fibonacci numbers.",
    "What does GPU stand for?",
    "Write one sentence about the Pacific Ocean.",
    "Convert 100 degrees Celsius to Fahrenheit.",
    "What is the largest planet in our solar system?",
    "Spell the word 'necessary'.",
    "What is 2 to the power of 10?",
    "Name the four cardinal directions.",
    "In one sentence, what does an operating system do?",
    "What is the square root of 144?",
    "List three programming languages.",
    "Who wrote the play Hamlet?",
    "What is the boiling point of water at sea level in Celsius?",
    "Summarize in one sentence: the water cycle moves water between the ocean, "
    "atmosphere and land through evaporation, condensation and precipitation.",
    "What is the difference between a list and a tuple in Python?",
    "How many continents are there?",
    "What is the speed of light in a vacuum, approximately?",
    "Give one example of a renewable energy source.",
    "What is 15% of 200?",
    "Explain recursion in one sentence.",
]



def _bf16_ulp(x: float) -> float:
    """Spacing between adjacent bfloat16 values at magnitude |x|.

    bf16 has 8 mantissa bits, so ULP = 2**(exponent-7). Two distinct bf16 logits can
    never be closer than this, which is why a near-tie threshold must be expressed in
    ULP rather than as an absolute value.
    """
    import math

    ax = abs(float(x))
    if ax == 0.0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(ax)) - 7)


def minisgl_generate(prompt_text: str, max_tokens: int) -> list[int]:
    """Ask the server to continue the ALREADY-RENDERED `prompt_text` greedily.

    Sends `prompt`, not `messages`, so TokenizeManager takes the `else` branch and skips
    its own apply_chat_template (tokenizer/tokenize.py:19). Passing `messages` would have
    each side render independently -- fine for Qwen, but gpt-oss's harmony template
    injects a current-date system line and a reasoning-effort header, so a rendering
    difference would look like a model bug. One rendering, both sides.
    """
    body = json.dumps(
        {
            "model": "q",
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": 0,
            "return_token_ids": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    ids: list[int] = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
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
                tids = choice.get("token_ids")
                if tids:
                    ids.extend(int(t) for t in tids)
    return ids


def build_reference(prompts, tok, max_tokens: int) -> list[dict]:
    """Run HF greedily and record, per step, the ids AND the top-5 logits.

    Capturing the top-5 during `generate` rather than recomputing them later is what makes
    the near-tie test below sound. The previous version re-ran a full forward pass over the
    prefix at the divergence point, but HF's own KV-cached decode and a fresh full forward
    DISAGREE at few-ULP ties: observed at one position, the recompute ranked minisgl's token
    #1 (28.3750) while HF's cached decode had emitted #2 (28.0000), 3 ULP apart. That forced
    an awkward symmetric test to avoid blaming the port for the reference's own instability.
    `output_scores=True` returns the very logits each token was argmax'd from, so the
    reference is now self-consistent by construction and no second forward pass is needed.
    """
    from transformers import AutoModelForCausalLM

    print("loading HF reference model (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=HF_DEVICE
    )
    model.eval()
    # HF_DEVICE may be "auto" (sharded over several cards), in which case there is no
    # single model device to send inputs to -- the embedding's device is the entry point.
    in_dev = model.get_input_embeddings().weight.device
    print(f"      hf entry device={in_dev}", flush=True)

    records = []
    for i, prompt in enumerate(prompts):
        # one rendering, used by BOTH sides
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Match the server exactly: it calls tokenizer.encode(prompt) with DEFAULT
        # add_special_tokens (tokenizer/tokenize.py:28). Forcing False here while the
        # server leaves it True would prepend a BOS on one side only and shift every
        # position -- a false FAIL that looks like a real divergence at token 0.
        input_ids = tok(text, return_tensors="pt").input_ids

        with torch.no_grad():
            out = model.generate(
                input_ids.to(in_dev),
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        hf_ids = out.sequences[0, input_ids.shape[1]:].tolist()
        # do_sample=False with no top_k/top_p means no logits processor rewrote these,
        # so scores[j] are the raw logits token j was chosen from.
        top5 = []
        for step in out.scores:
            v, ix = torch.topk(step[0].float(), 5)
            top5.append({"ids": ix.tolist(), "vals": v.tolist()})
        records.append({"prompt": prompt, "text": text,
                        "n_prompt_ids": int(input_ids.shape[1]),
                        "hf_ids": hf_ids, "top5": top5})
        print(f"  hf p{i:02d}: {len(hf_ids)} tokens", flush=True)
    return records


def compare(records: list[dict], max_tokens: int) -> int:
    exact = 0
    near_tie = 0
    real_diff = []

    for i, rec in enumerate(records):
        text, hf_ids, top5 = rec["text"], rec["hf_ids"], rec["top5"]
        if i == 0:
            print(f"      rendered prompt p00: {rec['n_prompt_ids']} ids "
                  f"(both sides encode this same string)", flush=True)

        got_ids = minisgl_generate(text, max_tokens)

        n = min(len(hf_ids), len(got_ids))
        first_bad = next((j for j in range(n) if hf_ids[j] != got_ids[j]), None)
        if first_bad is None and len(hf_ids) == len(got_ids):
            exact += 1
            verdict = "EXACT"
            detail = f"{len(got_ids)} tokens"
        else:
            pos = first_bad if first_bad is not None else n
            if pos >= len(top5):
                verdict = "DIFF"
                real_diff.append((i, pos, float("nan")))
                print(f"[{verdict:8}] p{i:02d}: length mismatch past HF's last step "
                      f"(len hf={len(hf_ids)} got={len(got_ids)})", flush=True)
                continue
            tv, ti = top5[pos]["vals"], top5[pos]["ids"]
            gap = tv[0] - tv[1]
            # Near-tie must be judged in units of bf16 ULP, not as an absolute
            # threshold. bf16 has an 8-bit mantissa, so at |logit| ~ 20-40 one ULP is
            # already 0.125-0.25 and two distinct bf16 logits CANNOT be closer than
            # that. An absolute 1e-2 threshold is below the representable resolution,
            # so it would classify nothing as a near-tie -- it is unsatisfiable, not
            # strict.
            ulp = _bf16_ulp(tv[0])
            got_tok = got_ids[pos] if pos < len(got_ids) else None
            # These scores ARE the ones hf_ids[pos] was argmax'd from, so HF's own token
            # is top-1 by construction and only minisgl's needs checking. That makes the
            # test one-sided -- which is correct here, not a weakening: the two-sided
            # version existed solely to absorb the recompute-vs-cache disagreement that
            # capturing scores during generate has now eliminated.
            d_got = None
            if got_tok is not None and got_tok in ti:
                d_got = tv[0] - tv[ti.index(got_tok)]
            tie = d_got is not None and d_got <= 4 * ulp
            if tie:
                near_tie += 1
                verdict = "NEAR-TIE"
            else:
                verdict = "DIFF"
                real_diff.append((i, pos, gap))
            # How far outside the window, in ULP. `gap` (top1-top2) does NOT answer this:
            # the classification is on d_got (top1 minus OUR token), and a DIFF at 4.1 ULP
            # is a different claim from one at 100 ULP or one where our token is not in
            # the top-5 at all. Without this the pass/fail is reported without its
            # effect size, which is what made the earlier "FAIL but marginal" hard to
            # act on. None = our token was outside HF's top-5.
            d_ulp = None if d_got is None else d_got / ulp
            a = hf_ids[pos] if pos < len(hf_ids) else None
            b = got_ids[pos] if pos < len(got_ids) else None
            d_str = "our_tok_outside_top5" if d_ulp is None else f"d_got={d_ulp:.1f}ULP"
            detail = (
                f"first divergence @{pos}: hf={a} got={b} top1-top2_gap={gap:.4f} "
                f"{d_str} (len hf={len(hf_ids)} got={len(got_ids)})"
            )
        print(f"[{verdict:8}] p{i:02d}: {detail}", flush=True)
    return exact, near_tie, real_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="number of prompts")
    ap.add_argument("--max-tokens", type=int, default=64)
    # Two-phase mode, for a model too large to hold alongside the server. gpt-oss-120b in
    # bf16 is ~234 GB and the TP4 server already reserves 90% of all four cards, so the
    # reference cannot be resident at the same time on a 4-card budget.
    ap.add_argument("--hf-out", help="phase A: dump the HF reference here, no server needed")
    ap.add_argument("--hf-in", help="phase B: compare against a dumped reference, no HF load")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = PROMPTS[: args.n]
    print(f"T-10: {len(prompts)} prompts, greedy, max_new_tokens={args.max_tokens}")
    print(f"      model={MODEL}")
    print(f"      server=127.0.0.1:{PORT}   hf_device={HF_DEVICE}\n", flush=True)

    if args.hf_in:
        with open(args.hf_in) as f:
            blob = json.load(f)
        if blob["model"] != MODEL:
            raise SystemExit(f"reference is for {blob['model']}, MODEL is {MODEL}")
        if blob["max_tokens"] != args.max_tokens:
            raise SystemExit(
                f"reference was generated with max_tokens={blob['max_tokens']}, "
                f"asked for {args.max_tokens}")
        records = blob["records"][: args.n]
        print(f"loaded {len(records)} reference records from {args.hf_in}", flush=True)
    else:
        records = build_reference(prompts, tok, args.max_tokens)

    if args.hf_out:
        with open(args.hf_out, "w") as f:
            json.dump({"model": MODEL, "max_tokens": args.max_tokens,
                       "records": records}, f)
        print(f"\nwrote reference for {len(records)} prompts to {args.hf_out}")
        print("phase A done -- free the cards, start the server, then re-run with --hf-in")
        return 0

    exact, near_tie, real_diff = compare(records, args.max_tokens)
    # len(records), not len(prompts): a dumped reference may hold fewer than --n.
    total = len(records)
    pct = 100.0 * exact / total
    print(f"\n=== T-10 summary ===")
    print(f"  exact match      : {exact}/{total}  ({pct:.1f}%)")
    print(f"  near-tie diverge : {near_tie}   (minisgl's token within 4 bf16 ULP of HF's top-1)")
    print(f"  real divergence  : {len(real_diff)}")
    for i, pos, gap in real_diff:
        print(f"      p{i:02d} @{pos} gap={gap:.4f}")
    # Gate: >=99% exact, and every mismatch explainable as a near-tie.
    ok = pct >= 99.0 or (exact + near_tie == total)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(gate: >=99% exact, or all mismatches near-ties)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
