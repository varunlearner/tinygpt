"""Measure what your GPU/iGPU actually delivers: fp32/fp16 matmul TFLOPS and memory copy bandwidth.

    python scripts/probe_xpu.py          # auto: cuda > xpu > cpu
    python scripts/probe_xpu.py xpu

Record the numbers in docs/ENVIRONMENT.md; they feed the roofline analysis (E6).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.utils import device_name, measure_copy_bandwidth_gbs, resolve_device, sync  # noqa: E402


def bench(fn, dev, n=20):
    for _ in range(3):
        fn()
    sync(dev)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / n


def main() -> None:
    dev = resolve_device(sys.argv[1] if len(sys.argv) > 1 else "auto")
    print(f"torch {torch.__version__} | device: {dev} ({device_name(dev)})")
    N = 4096 if dev.type != "cpu" else 1024
    for dt in (torch.float32, torch.float16):
        if dev.type == "cpu" and dt == torch.float16:
            continue
        a = torch.randn(N, N, device=dev, dtype=dt)
        b = torch.randn(N, N, device=dev, dtype=dt)
        print(f"{str(dt):15s} matmul: {2 * N ** 3 / bench(lambda: a @ b, dev) / 1e12:7.2f} TFLOPS")
    print(f"copy bandwidth : {measure_copy_bandwidth_gbs(dev):7.1f} GB/s")
    print("\nHow to read: if fp16 TFLOPS >> fp32, the device uses matrix engines; if ~1-2x, it does not "
          "(only affects absolute iGPU numbers).")


if __name__ == "__main__":
    main()
