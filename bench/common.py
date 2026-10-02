"""Shared benchmark utilities: prompts, timing, result files."""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from tinygpt.utils import sync


def get_prompts(data_dir: str, n: int, prompt_len: int, seed: int = 0) -> List[List[int]]:
    """A FIXED set of prompts (first `prompt_len` tokens of validation stories), cached on disk."""
    path = os.path.join(data_dir, f"bench_prompts_n{n}_len{prompt_len}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    eos_pos = np.flatnonzero(np.asarray(val) == meta["eos_id"])
    starts = np.concatenate([[0], eos_pos[:-1] + 1])
    ends = eos_pos
    ok = [int(s) for s, e in zip(starts, ends) if e - s >= prompt_len]
    if not ok:
        raise RuntimeError("no validation story is long enough for the requested prompt length")
    rng = np.random.default_rng(seed)
    pick = rng.choice(ok, size=n, replace=len(ok) < n)
    prompts = [val[s:s + prompt_len].astype(int).tolist() for s in pick]
    with open(path, "w") as f:
        json.dump(prompts, f)
    return prompts


def timed(fn: Callable[[], object], warmup: int, runs: int, device) -> List[float]:
    """Wall-clock seconds per call, with device sync before/after every timed call."""
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append(time.perf_counter() - t0)
    return out


def summarize(times: List[float]) -> Dict[str, float]:
    a = np.asarray(times, dtype=np.float64)
    return {"median": float(np.median(a)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def save_result(experiment: str, env: dict, config: dict, rows: list, out_dir: str = "results",
                extra: dict | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{experiment}_{env['device']}_{stamp}.json")
    obj = {"experiment": experiment, "env": env, "config": config, "rows": rows}
    if extra:
        obj.update(extra)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[bench] wrote {path}")
    return path


def iqr_warning(s: Dict[str, float], label: str) -> str:
    rel = (s["p75"] - s["p25"]) / s["median"] if s["median"] else 0.0
    return f"  !! IQR is {100 * rel:.0f}% of the median for {label}: check for background load / throttling" if rel > 0.10 else ""
