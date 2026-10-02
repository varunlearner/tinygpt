"""Static, preallocated KV cache with O(1) rollback.

Layout: k, v have shape [n_layers, batch, n_kv_heads, max_len, head_dim].
`length` is the number of valid positions (shared by every sequence in the batch).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class KVCache:
    def __init__(self, n_layers: int, batch_size: int, n_kv_heads: int, max_len: int, head_dim: int,
                 dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        shape = (n_layers, batch_size, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.max_len = max_len

    @classmethod
    def for_model(cls, model, batch_size: int = 1, max_len: Optional[int] = None) -> "KVCache":
        cfg = model.cfg
        p = next(model.parameters())
        return cls(cfg.n_layers, batch_size, cfg.n_kv_heads, max_len or cfg.max_seq_len, cfg.head_dim,
                   dtype=p.dtype, device=p.device)

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V at [length : length+T] and return views over [0 : length+T].

        `length` is NOT advanced here; GPT.forward calls advance() once after all layers ran.
        """
        t = k_new.shape[2]
        start = self.length
        end = start + t
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {end} > max_len {self.max_len}")
        self.k[layer, :, :, start:end] = k_new
        self.v[layer, :, :, start:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def truncate(self, n: int) -> None:
        """Roll back to n valid positions. O(1): no data is moved (stale entries get overwritten)."""
        self.length = min(self.length, n)

    def reset(self) -> None:
        self.length = 0

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2
