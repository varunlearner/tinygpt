"""Sampling: logits -> probability distribution -> token.

`probs_from_logits` is the ONE place where temperature / top-k / top-p are applied, and it is used
for both the draft and the target model in speculative decoding (their distributions must be
processed identically for the exactness guarantee to hold).

temperature == 0 returns a one-hot argmax distribution, so greedy decoding is just a special case
and the speculative accept/reject rule reduces to "accept iff the argmaxes agree".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingConfig:
    temperature: float = 1.0     # 0 => greedy
    top_k: int = 0               # 0 => disabled
    top_p: float = 1.0           # 1.0 => disabled


def probs_from_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """logits [..., V] -> probabilities [..., V] (float32)."""
    logits = logits.float()
    if cfg.temperature <= 0:
        idx = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    logits = logits / cfg.temperature
    if cfg.top_k and cfg.top_k < logits.shape[-1]:
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if cfg.top_p < 1.0:
        sp, si = torch.sort(probs, dim=-1, descending=True)
        cum = sp.cumsum(dim=-1)
        remove = (cum - sp) > cfg.top_p          # keep the smallest set whose mass reaches top_p
        sp = sp.masked_fill(remove, 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return probs


def sample_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """probs [B, V] -> token ids [B, 1]."""
    return torch.multinomial(probs, 1, generator=generator)
