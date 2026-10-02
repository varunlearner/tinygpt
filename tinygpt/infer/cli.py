"""Generate text from a trained model with the naive, KV-cache or speculative decoder.

    python -m tinygpt.infer.cli --ckpt runs/target_28m/model.pt --prompt "Once upon a time" --mode kvcache
    python -m tinygpt.infer.cli --ckpt runs/target_28m/model.pt --draft runs/draft_4m/model.pt --mode speculative --k 4
"""
from __future__ import annotations

import argparse
import time

import torch

from tinygpt.infer.generate import generate_cached, generate_naive
from tinygpt.infer.sampling import SamplingConfig
from tinygpt.infer.speculative import generate_speculative
from tinygpt.model.io import load_model
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.utils import make_generator, resolve_device, sync, torch_dtype


def default_dtype(device: torch.device) -> str:
    return "fp16" if device.type in ("cuda", "xpu") else "fp32"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--tokenizer", default="data/tok/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--mode", choices=["naive", "kvcache", "speculative"], default="kvcache")
    ap.add_argument("--k", type=int, default=4, help="draft tokens per round (speculative)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    dtype = default_dtype(dev) if args.dtype == "auto" else args.dtype
    model = load_model(args.ckpt, dev, torch_dtype(dtype))
    tok = BPETokenizer.load(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt)], device=dev)
    n_new = min(args.max_new_tokens, model.cfg.max_seq_len - ids.shape[1])
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    gen = make_generator(dev, args.seed)

    stats = None
    sync(dev)
    t0 = time.perf_counter()
    if args.mode == "naive":
        out = generate_naive(model, ids, n_new, sampling, tok.eos_id, gen)
    elif args.mode == "kvcache":
        out = generate_cached(model, ids, n_new, sampling, tok.eos_id, gen)
    else:
        if not args.draft:
            raise SystemExit("--draft is required for --mode speculative")
        draft = load_model(args.draft, dev, torch_dtype(dtype))
        out, stats = generate_speculative(model, draft, ids, n_new, args.k, sampling, tok.eos_id, gen)
    sync(dev)
    dt = time.perf_counter() - t0

    new = out.shape[1] - ids.shape[1]
    print(tok.decode(out[0].tolist()))
    print(f"\n[{args.mode}] {new} tokens in {dt:.2f}s = {new / dt:.1f} tok/s  (device={dev}, dtype={dtype})")
    if stats:
        print(f"[speculative] acceptance rate {stats.acceptance_rate:.2f} | {stats.mean_tokens_per_round:.2f} tokens/round "
              f"| {stats.rounds} rounds")


if __name__ == "__main__":
    main()
