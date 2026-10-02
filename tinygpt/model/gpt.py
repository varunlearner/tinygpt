"""Llama-style decoder-only transformer: RoPE, RMSNorm (pre-norm), SwiGLU, no biases, tied embeddings.

The same forward pass is used for training, naive generation, cached generation and speculative
decoding.  Pass a KVCache to decode incrementally; RoPE positions then start at cache.length.

RoPE uses the "rotate-half" layout, identical to Hugging Face Llama, so export needs no weight permutation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.model.config import GPTConfig
from tinygpt.model.kv_cache import KVCache


def build_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dt)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        d = cfg.d_model
        self.wq = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache], layer_idx: int, start_pos: int) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.wv(x).view(B, T, Hkv, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)      # -> [B, Hkv, start_pos+T, D]
        if self.n_rep > 1:                            # grouped-query attention: share KV heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if T == 1:
            y = F.scaled_dot_product_attention(q, k, v)                       # sees everything cached
        elif start_pos == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # plain causal
        else:                                                                 # chunk appended to a cache
            mask = torch.ones(T, start_pos + T, dtype=torch.bool, device=x.device).tril(diagonal=start_pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(B, T, H * D))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache, layer_idx, start_pos):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx, start_pos)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)       # GPT-2 style scaled init on residual projections
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=scale)
            nn.init.normal_(blk.mlp.w_down.weight, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _apply(self, fn, recurse=True):
        # Keep RoPE tables in fp32 even after .half()/.bfloat16(): rebuild them on the target device.
        out = super()._apply(fn, recurse)
        dev = self.rope_cos.device
        cos, sin = build_rope(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)
        self.rope_cos, self.rope_sin = cos.to(dev), sin.to(dev)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, cache: Optional[KVCache] = None, start_pos: int = 0,
                logits_last: Optional[int] = None) -> torch.Tensor:
        """idx: [B, T] token ids -> logits [B, T, V] (or [B, logits_last, V]).

        With a cache, positions start at cache.length and the cache is advanced by T.
        """
        B, T = idx.shape
        if cache is not None:
            start_pos = cache.length
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence too long: {start_pos}+{T} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, cache, i, start_pos)
        if cache is not None:
            cache.advance(T)
        x = self.norm(x)
        if logits_last is not None:
            x = x[:, -logits_last:]
        return self.lm_head(x)
