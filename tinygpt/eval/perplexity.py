"""Full validation-set evaluation: loss (nats/token), perplexity, bits-per-byte.

    python -m tinygpt.eval.perplexity --ckpt runs/target_28m/model.pt --data-dir data/shards
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from tinygpt.data.loader import TokenData
from tinygpt.model.io import load_model
from tinygpt.utils import resolve_device, torch_dtype


@torch.no_grad()
def evaluate(ckpt: str, data_dir: str, device: str = "auto", dtype: str = "fp32", batch_size: int = 32) -> dict:
    dev = resolve_device(device)
    model = load_model(ckpt, dev, torch_dtype(dtype))
    data = TokenData(data_dir)
    ctx = model.cfg.max_seq_len
    nll, n_tok = 0.0, 0
    for x, y in data.sequential_val(batch_size, ctx, dev):
        logits = model(x)
        nll += F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        n_tok += y.numel()
    loss = nll / n_tok
    tokens_per_byte = data.meta["val_tokens"] / data.meta["val_bytes"]
    return {
        "ckpt": ckpt, "params": model.num_params(), "val_tokens_evaluated": n_tok, "dtype": dtype,
        "val_loss_nats_per_token": loss, "perplexity": math.exp(loss),
        "bits_per_byte": loss / math.log(2) * tokens_per_byte,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    res = evaluate(args.ckpt, args.data_dir, args.device, args.dtype)
    print(json.dumps(res, indent=2))
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))) or "model"
    with open(os.path.join(args.out, f"eval_{name}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
