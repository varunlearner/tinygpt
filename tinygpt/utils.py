"""Small shared helpers: seeding, device handling, timing sync, environment report."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def make_generator(device: torch.device | str, seed: int):
    """Seeded RNG on `device`; falls back to the global RNG if the backend has no device generators (some XPU builds)."""
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        torch.manual_seed(seed)
        return None


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks cuda > xpu > cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if dev.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU requested but torch.xpu is not available (install the XPU build of PyTorch)")
    return dev


def sync(device: torch.device | str) -> None:
    """Block until all queued work on the device has finished (needed for honest timing)."""
    t = torch.device(device).type
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "xpu":
        torch.xpu.synchronize()


def device_name(device: torch.device | str) -> str:
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(0)
    if dev.type == "xpu":
        return torch.xpu.get_device_name(0)
    try:  # nicer CPU name on Linux
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def peak_memory_bytes(device: torch.device | str) -> Optional[int]:
    dev = torch.device(device)
    if dev.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if dev.type == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return int(torch.xpu.max_memory_allocated())
    return None


def reset_peak_memory(device: torch.device | str) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif dev.type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()


def best_train_dtype(device: torch.device) -> str:
    """fp16 on GPUs without real bf16 (e.g. T4), bf16 where supported, fp32 on CPU."""
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return "bf16" if major >= 8 else "fp16"
    if device.type == "xpu":
        return "bf16"
    return "fp32"


def torch_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16,
            "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[name]


@torch.no_grad()
def measure_copy_bandwidth_gbs(device: torch.device | str, mbytes: int = 256, iters: int = 10) -> float:
    """Measured memory bandwidth (GB/s) using a device-to-device copy. Used for roofline / MBU."""
    dev = torch.device(device)
    x = torch.empty(mbytes * 2**20 // 4, device=dev, dtype=torch.float32)
    x.fill_(1.0)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    sync(dev)
    dt = (time.perf_counter() - t0) / iters
    return 2 * x.numel() * 4 / dt / 1e9  # read + write


def git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_report(device: torch.device | str = "cpu", dtype: str = "fp32",
               measure_bandwidth: bool = False, power_mode: str = "unspecified") -> dict:
    dev = torch.device(device)
    rep = {
        "device": dev.type,
        "device_name": device_name(dev),
        "dtype": dtype,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
        "power_mode": power_mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if dev.type == "cuda":
        rep["cuda"] = torch.version.cuda
        rep["capability"] = list(torch.cuda.get_device_capability())
        rep["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    if measure_bandwidth:
        try:
            rep["copy_bandwidth_gbs"] = round(measure_copy_bandwidth_gbs(dev), 2)
        except Exception as e:  # pragma: no cover
            rep["copy_bandwidth_gbs"] = None
            rep["bandwidth_error"] = str(e)
    return rep


if __name__ == "__main__":
    print(json.dumps(env_report("cpu", measure_bandwidth=True), indent=2))
