"""Memmap-backed token loader with random-window sampling (nanoGPT style)."""
from __future__ import annotations

import json
import os
from typing import Iterator, Tuple

import numpy as np
import torch


class TokenData:
    def __init__(self, data_dir: str):
        self.dir = data_dir
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        train_path = os.path.join(data_dir, "train.bin")   # optional: evaluation / benchmarks only need val.bin
        self.train = np.memmap(train_path, dtype=np.uint16, mode="r") if os.path.exists(train_path) else None
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def _windows(self, data: np.ndarray, starts: np.ndarray, ctx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = starts[:, None] + np.arange(ctx + 1, dtype=np.int64)[None, :]
        chunk = torch.from_numpy(data[idx].astype(np.int64))
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

    def get_batch(self, split: str, batch_size: int, ctx: int, rng: np.random.Generator,
                  device: torch.device | str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if data is None:
            raise FileNotFoundError(f"{self.dir}/train.bin not found (only needed for training)")
        starts = rng.integers(0, len(data) - ctx - 1, size=batch_size)
        x, y = self._windows(data, starts, ctx)
        dev = torch.device(device)
        if dev.type == "cuda":
            return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
        return x.to(dev), y.to(dev)

    def val_batches(self, n_batches: int, batch_size: int, ctx: int, device: torch.device | str = "cpu",
                    seed: int = 12345) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """A FIXED set of validation windows (same seed -> same windows) so val loss is comparable."""
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            yield self.get_batch("val", batch_size, ctx, rng, device)

    def sequential_val(self, batch_size: int, ctx: int, device: torch.device | str = "cpu"
                       ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows covering the whole validation set (for the final evaluation)."""
        n_windows = (len(self.val) - 1) // ctx
        starts_all = np.arange(n_windows, dtype=np.int64) * ctx
        for i in range(0, n_windows, batch_size):
            x, y = self._windows(self.val, starts_all[i:i + batch_size], ctx)
            yield x.to(device), y.to(device)
