"""Save / load model weights (+ config) in a single .pt file."""
from __future__ import annotations

import os
from typing import Optional

import torch

from tinygpt.model.config import GPTConfig
from tinygpt.model.gpt import GPT


def clean_state_dict(sd: dict) -> dict:
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


def save_model(model: GPT, path: str, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    obj = {"model_config": raw.cfg.to_dict(), "model": clean_state_dict(raw.state_dict())}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_model(path: str, device: torch.device | str = "cpu", dtype: Optional[torch.dtype] = None) -> GPT:
    """Works for both final model files and training checkpoints."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(obj["model_config"])
    model = GPT(cfg)
    model.load_state_dict(clean_state_dict(obj["model"]))
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model.eval()
