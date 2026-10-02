"""Speculative decoding (Leviathan et al. 2022 / Chen et al. 2023), batch size 1.

Each round:
  1. the draft model q proposes k tokens autoregressively (cheap, own KV cache);
  2. the target model p scores [last_token, x_1..x_k] in ONE forward pass -> p_1..p_{k+1};
  3. accept x_i with probability min(1, p_i(x_i)/q_i(x_i)); at the first rejection resample from
     normalize(max(0, p_i - q_i)) and stop; if all k are accepted sample a bonus token from p_{k+1}.

This leaves the output distribution EXACTLY equal to sampling from the target alone.

Cache invariant: every cache holds K/V for seq[0 : cache.length]; the newest token is not cached yet.
  * target cache length at round start = len(seq) - 1
  * draft cache can be shorter (after "all k accepted" it never saw x_k) -> it catches up first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from tinygpt.infer.sampling import SamplingConfig, probs_from_logits, sample_from_probs
from tinygpt.model.gpt import GPT
from tinygpt.model.kv_cache import KVCache


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0            # draft tokens proposed
    accepted: int = 0            # draft tokens accepted
    tested: int = 0              # positions at which an accept/reject test was made
    generated: int = 0           # new tokens produced
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Estimate of alpha: accepted / tested (unbiased under early stopping)."""
        return self.accepted / self.tested if self.tested else 0.0

    @property
    def mean_tokens_per_round(self) -> float:
        return self.generated / self.rounds if self.rounds else 0.0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["acceptance_rate"] = self.acceptance_rate
        d["mean_tokens_per_round"] = self.mean_tokens_per_round
        return d


@torch.no_grad()
def generate_speculative(target: GPT, draft: GPT, prompt: torch.Tensor, max_new_tokens: int, k: int,
                         sampling: SamplingConfig, eos_id: Optional[int] = None,
                         generator: Optional[torch.Generator] = None,
                         target_cache: Optional[KVCache] = None,
                         draft_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, SpecStats]:
    assert prompt.shape[0] == 1, "speculative decoding here supports batch size 1"
    assert target.cfg.vocab_size == draft.cfg.vocab_size, "draft and target must share a tokenizer"
    assert k >= 1
    dev = prompt.device
    T0 = prompt.shape[1]
    total_max = T0 + max_new_tokens
    if total_max > min(target.cfg.max_seq_len, draft.cfg.max_seq_len):
        raise ValueError("prompt + max_new_tokens exceeds the context length")

    tcache = target_cache or KVCache.for_model(target, 1)
    dcache = draft_cache or KVCache.for_model(draft, 1)
    tcache.reset()
    dcache.reset()

    seq = torch.empty(1, total_max, dtype=prompt.dtype, device=dev)   # device-side token buffer
    seq[:, :T0] = prompt
    n = T0                                                            # tokens currently in seq
    stats = SpecStats()

    # prefill everything except the last prompt token (it is fed at the start of the first round)
    if T0 > 1:
        target(seq[:, :T0 - 1], cache=tcache, logits_last=1)
        stats.target_forwards += 1

    def finish(n_tok: int) -> Tuple[torch.Tensor, SpecStats]:
        stats.generated = n_tok - T0
        return seq[:, :n_tok].clone(), stats

    while n - T0 < max_new_tokens:
        remaining = max_new_tokens - (n - T0)
        k_eff = min(k, remaining - 1)

        if k_eff <= 0:   # only one token left: plain target step
            logits = target(seq[:, tcache.length:n], cache=tcache, logits_last=1)[:, -1]
            stats.target_forwards += 1
            tok = sample_from_probs(probs_from_logits(logits, sampling), generator)
            seq[:, n] = tok[:, 0]
            n += 1
            stats.rounds += 1
            break

        # ---- 1. draft proposes k_eff tokens ------------------------------------------------
        q_probs, xs = [], []
        logits = draft(seq[:, dcache.length:n], cache=dcache, logits_last=1)[:, -1]   # catch-up + q_1
        stats.draft_forwards += 1
        for i in range(k_eff):
            q = probs_from_logits(logits, sampling)                                   # [1, V]
            x = sample_from_probs(q, generator)                                       # [1, 1]
            q_probs.append(q[0])
            xs.append(x)
            if i < k_eff - 1:
                logits = draft(x, cache=dcache, logits_last=1)[:, -1]
                stats.draft_forwards += 1
        x_all = torch.cat(xs, dim=1)                                                  # [1, k_eff]
        q_all = torch.stack(q_probs, dim=0)                                           # [k_eff, V]

        # ---- 2. target verifies in one pass -------------------------------------------------
        verify_in = torch.cat([seq[:, n - 1:n], x_all], dim=1)                        # [1, k_eff+1]
        p_all = probs_from_logits(target(verify_in, cache=tcache)[0], sampling)       # [k_eff+1, V]
        stats.target_forwards += 1

        # ---- 3. accept / reject ------------------------------------------------------------
        ar = torch.arange(k_eff, device=dev)
        xi = x_all[0]
        ratio = p_all[ar, xi] / q_all[ar, xi].clamp_min(1e-30)
        u = torch.rand(k_eff, device=dev, generator=generator)
        accept = (u < ratio).tolist()
        n_acc = 0
        while n_acc < k_eff and accept[n_acc]:
            n_acc += 1

        if n_acc < k_eff:                                                             # rejection: residual
            resid = (p_all[n_acc] - q_all[n_acc]).clamp_min(0)
            resid = resid / resid.sum().clamp_min(1e-30)
            y = sample_from_probs(resid[None], generator)
            stats.tested += n_acc + 1
        else:                                                                         # all accepted: bonus
            y = sample_from_probs(p_all[k_eff][None], generator)
            stats.tested += k_eff
        stats.rounds += 1
        stats.proposed += k_eff
        stats.accepted += n_acc

        new = torch.cat([x_all[:, :n_acc], y], dim=1)                                 # [1, n_acc+1]
        new = new[:, :remaining]
        if eos_id is not None:
            hit = (new[0] == eos_id).nonzero()
            if len(hit):
                new = new[:, :int(hit[0]) + 1]
        m = new.shape[1]
        seq[:, n:n + m] = new
        n += m

        # ---- 4. roll both caches back to the accepted prefix --------------------------------
        tcache.truncate(n - 1)
        dcache.truncate(n - 1)
        if eos_id is not None and int(new[0, -1]) == eos_id:
            break

    return finish(n)
