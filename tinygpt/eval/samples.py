"""Generate a fixed, seeded set of samples (10 hand-written prompts + story openings from the val set)
so quality can be judged without cherry-picking.

    python -m tinygpt.eval.samples --ckpt runs/target_28m/model.pt --tokenizer data/tok/tokenizer.json \
        --data-dir data/shards --out results/samples/target_28m.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from tinygpt.data.loader import TokenData
from tinygpt.infer.generate import generate_cached
from tinygpt.infer.sampling import SamplingConfig
from tinygpt.model.io import load_model
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.utils import make_generator, resolve_device, torch_dtype

FIXED_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy named Tom found a shiny red ball in the garden.",
    "The little dog was very sad because",
    "Anna and her mom went to the park. They saw",
    "Ben wanted to build a big tower, but",
    "In a small town, there lived a kind old man who",
    "The cat and the bird were best friends. One day",
    "Mia was scared of the dark. Her dad said,",
    "It was a sunny day. Sam put on his hat and",
    "There was a magic tree in the forest. When Zoe touched it,",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tok/tokenizer.json")
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--out", default="results/samples/samples.md")
    ap.add_argument("--n-val-prompts", type=int, default=40)
    ap.add_argument("--val-prompt-tokens", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    model = load_model(args.ckpt, dev, torch_dtype(args.dtype))
    tok = BPETokenizer.load(args.tokenizer)
    prompts = list(FIXED_PROMPTS)
    if os.path.exists(os.path.join(args.data_dir, "meta.json")):
        data = TokenData(args.data_dir)
        eos_pos = np.flatnonzero(np.asarray(data.val[:2_000_000]) == tok.eos_id)
        starts = [0] + (eos_pos[:-1] + 1).tolist()
        for s in starts[:args.n_val_prompts]:
            prompts.append(tok.decode(data.val[s:s + args.val_prompt_tokens].astype(int).tolist()))

    cfg = SamplingConfig(temperature=args.temperature, top_k=args.top_k)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Samples: {args.ckpt}\n\ntemperature={args.temperature}, top_k={args.top_k}, seed=0 "
                f"(fixed prompts, no cherry-picking)\n\n")
        for i, p in enumerate(prompts):
            ids = torch.tensor([tok.encode(p)], device=dev)
            n_new = min(args.max_new_tokens, model.cfg.max_seq_len - ids.shape[1])
            g = make_generator(dev, i)
            out = generate_cached(model, ids, n_new, cfg, eos_id=tok.eos_id, generator=g)
            f.write(f"### {i + 1}. `{p}`\n\n{tok.decode(out[0].tolist())}\n\n---\n\n")
    print(f"wrote {len(prompts)} samples -> {args.out}")


if __name__ == "__main__":
    main()
