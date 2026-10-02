"""int8 weight-only quantization (per-output-channel, symmetric) + quality/speed report (experiment E7).

    python -m export.quantize --ckpt runs/target_28m/model.pt --data-dir data/shards --device cuda

NOTE: this is a *simulation* of weight-only int8: weights are stored as int8 (4x smaller than fp32, 2x than
fp16) and dequantized on the fly inside each Linear. Without a fused int8 kernel the dequantize step costs
extra memory traffic, so a speedup is NOT guaranteed - that is exactly what E7 measures. The val-loss delta is
the reliable part of the result.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.model.gpt import GPT


class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.detach().float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        self.register_buffer("qweight", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).to(weight.dtype))
        self.out_features, self.in_features = w.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        return F.linear(x, w)


def quantize_model_int8(model: GPT) -> GPT:
    """Return a copy with every attention/MLP Linear replaced by Int8Linear (embedding/LM head stay full precision)."""
    q = copy.deepcopy(model)
    for blk in q.blocks:
        for parent in (blk.attn, blk.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear(child.weight.data))
    return q.eval()


def linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, Int8Linear):
            total += m.qweight.numel() + m.scale.numel() * m.scale.element_size()
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
    return total


def main() -> None:
    from bench.common import save_result
    from bench.run import measure_decode_step
    from tinygpt.data.loader import TokenData
    from tinygpt.model.io import load_model
    from tinygpt.utils import env_report, resolve_device, torch_dtype

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    dtype = args.dtype if args.dtype != "auto" else ("fp16" if dev.type in ("cuda", "xpu") else "fp32")
    model = load_model(args.ckpt, dev, torch_dtype(dtype))
    qmodel = quantize_model_int8(model)
    data = TokenData(args.data_dir)

    @torch.no_grad()
    def val_loss(m: GPT) -> float:
        losses = []
        for x, y in data.val_batches(args.batches, 16, m.cfg.max_seq_len, dev):
            lg = m(x)
            losses.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.reshape(-1)).item())
        return sum(losses) / len(losses)

    l_fp, l_q = val_loss(model), val_loss(qmodel)
    t_fp, t_q = measure_decode_step(model, dev), measure_decode_step(qmodel, dev)
    rows = [{"variant": dtype, "val_loss": l_fp, "linear_weight_MB": linear_weight_bytes(model) / 1e6, "decode_step_ms": t_fp * 1e3},
            {"variant": "int8-weight-only", "val_loss": l_q, "linear_weight_MB": linear_weight_bytes(qmodel) / 1e6,
             "decode_step_ms": t_q * 1e3}]
    print(json.dumps(rows, indent=2))
    print(f"val-loss delta: {l_q - l_fp:+.4f} nats ({100 * (math.exp(l_q - l_fp) - 1):+.2f}% perplexity) | "
          f"decode step speedup: {t_fp / t_q:.2f}x")
    save_result("E7", env_report(dev, dtype), vars(args), rows, args.out,
                {"val_loss_delta": l_q - l_fp, "decode_speedup": t_fp / t_q})


if __name__ == "__main__":
    main()
