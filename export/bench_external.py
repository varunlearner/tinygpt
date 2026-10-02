"""E8: compare our engine with Hugging Face `generate` and OpenVINO on the same prompt (batch 1, greedy).

    python -m export.bench_external --native runs/target_28m/model.pt --hf-dir export/hf \
        --ov-dir export/ov --ov-device GPU --new-tokens 128
"""
from __future__ import annotations

import argparse

import torch

from bench.common import get_prompts, save_result, summarize, timed
from tinygpt.infer.generate import generate_cached
from tinygpt.infer.sampling import SamplingConfig
from tinygpt.model.io import load_model
from tinygpt.utils import env_report, resolve_device, torch_dtype


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--ov-dir", default=None)
    ap.add_argument("--ov-device", default="CPU", help="CPU | GPU (Arc iGPU) | NPU")
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--device", default="cpu", help="device for native + HF PyTorch")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    dtype = "fp16" if dev.type in ("cuda", "xpu") else "fp32"
    prompt_ids = get_prompts(args.data_dir, 1, args.prompt_len)[0]
    ids = torch.tensor([prompt_ids], device=dev)
    rows = []

    def record(name: str, fn) -> None:
        s = summarize(timed(fn, args.warmup, args.runs, dev))
        rows.append({"engine": name, "seconds": s, "tokens_per_s": args.new_tokens / s["median"]})
        print(f"[E8] {name:28s} {args.new_tokens / s['median']:9.1f} tok/s")

    native = load_model(args.native, dev, torch_dtype(dtype))
    greedy = SamplingConfig(temperature=0.0)
    record("tinygpt (KV cache, eager)", lambda: generate_cached(native, ids, args.new_tokens, greedy))

    if args.hf_dir:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch_dtype(dtype)).to(dev).eval()
        gen = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)
        record("HF transformers generate", lambda: hf.generate(ids, attention_mask=torch.ones_like(ids), **gen))

    if args.ov_dir:
        from optimum.intel import OVModelForCausalLM
        ov = OVModelForCausalLM.from_pretrained(args.ov_dir, device=args.ov_device)
        gen = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)
        cpu_ids = ids.cpu()
        record(f"OpenVINO ({args.ov_device})", lambda: ov.generate(cpu_ids, attention_mask=torch.ones_like(cpu_ids), **gen))

    save_result("E8", env_report(dev, dtype), vars(args), rows, args.out)


if __name__ == "__main__":
    main()
