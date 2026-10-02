"""Baseline generators: naive (recompute everything every step) and KV-cached."""
from __future__ import annotations

from typing import Optional

import torch

from tinygpt.infer.sampling import SamplingConfig, probs_from_logits, sample_from_probs
from tinygpt.model.gpt import GPT
from tinygpt.model.kv_cache import KVCache


def _finish(tokens: torch.Tensor, finished: torch.Tensor, nxt: torch.Tensor, eos_id: Optional[int]):
    """Mark sequences that hit EOS; keep emitting EOS for finished rows."""
    if eos_id is None:
        return nxt, finished
    nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
    finished = finished | (nxt[:, 0] == eos_id)
    return nxt, finished


@torch.no_grad()
def generate_naive(model: GPT, prompt: torch.Tensor, max_new_tokens: int, sampling: SamplingConfig,
                   eos_id: Optional[int] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Every step runs the full forward pass over the whole sequence: O(n^2) total work."""
    tokens = prompt
    finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
    for _ in range(max_new_tokens):
        logits = model(tokens, logits_last=1)[:, -1]
        nxt = sample_from_probs(probs_from_logits(logits, sampling), generator)
        nxt, finished = _finish(tokens, finished, nxt, eos_id)
        tokens = torch.cat([tokens, nxt], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
    return tokens


@torch.no_grad()
def generate_cached(model: GPT, prompt: torch.Tensor, max_new_tokens: int, sampling: SamplingConfig,
                    eos_id: Optional[int] = None, generator: Optional[torch.Generator] = None,
                    cache: Optional[KVCache] = None) -> torch.Tensor:
    """Prefill the prompt once, then decode one token per step reusing cached K/V."""
    B, T0 = prompt.shape
    if cache is None:
        cache = KVCache.for_model(model, batch_size=B)
    else:
        cache.reset()
    out = torch.empty(B, T0 + max_new_tokens, dtype=prompt.dtype, device=prompt.device)
    out[:, :T0] = prompt
    finished = torch.zeros(B, dtype=torch.bool, device=prompt.device)

    logits = model(prompt, cache=cache, logits_last=1)[:, -1]            # prefill
    n = 0
    for n in range(max_new_tokens):
        nxt = sample_from_probs(probs_from_logits(logits, sampling), generator)
        nxt, finished = _finish(out, finished, nxt, eos_id)
        out[:, T0 + n] = nxt[:, 0]
        if eos_id is not None and bool(finished.all()):
            return out[:, :T0 + n + 1]
        if n < max_new_tokens - 1:
            logits = model(nxt, cache=cache, logits_last=1)[:, -1]       # decode
    return out
