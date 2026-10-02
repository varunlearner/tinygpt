# Environment

**Last updated:** 2026-09-30 · Free-tier limits change without notice — re-check them in your own account before planning long runs.

> Exact commands for every step: [`RUNBOOK.md`](RUNBOOK.md). Helper scripts: `scripts/probe_xpu.py`, `scripts/env_report.py`.

## 1. Compute matrix

| Machine | Specs | Role |
|---|---|---|
| **Kaggle notebook** | T4 (×2 available) or P100 · ~13 GB system RAM · free GPU quota that Kaggle sets and shows in account settings (commonly reported ≈30 h/week) · sessions up to ~12 h | Real training runs, T4 benchmarks |
| **Colab free tier** | GPU access without a published quota · idle timeouts · sessions up to ~12 h | Backup only |
| **Laptop** | Intel Core Ultra 7 155H (6P + 8E + 2LP cores), Arc iGPU (8 Xe-cores, 128 EUs), 16 GB RAM (15.4 usable, shared with iGPU), Windows, ~640 GB free disk | Development, smoke tests, CPU/XPU/OpenVINO benchmarks |

Prefer the **T4** over the P100 on Kaggle: the T4 has fp16 tensor cores; the P100 does not.

## 2. Kaggle workflow

1. **Data:** upload `train.bin`, `val.bin`, `tokenizer.json` as a private Kaggle Dataset (create a new dataset version when they change).
2. **Code:** enable notebook internet, `git clone` the repo in the first cell (or attach it as a dataset).
3. **Long runs:** use **Save & Run All (Commit)** so training continues in the background without the browser open. Interactive GPU sessions have short idle timeouts.
4. **Checkpoints:** write to `/kaggle/working/` every ~15 min / 500 steps (model, optimizer, scaler, RNG, data position). Keep the last 2–3 and the best.
5. **Persist:** after each session, save `/kaggle/working` output as a new Dataset version and/or push to Hugging Face Hub (token via Kaggle Secrets). **Never leave the only copy of the weights in a Kaggle session.**
6. **Resume:** next session, attach the checkpoint dataset and run with `--resume`.
7. **Track quota:** note remaining GPU hours at the start of each session in `LEARNINGS.md`.
8. **Secrets:** W&B / HF tokens via Kaggle Secrets, never committed.

Two GPUs: start with one. DDP over 2×T4 is a P2 item.

## 3. Laptop workflow

### 3.1 One-time setup (Windows)

1. Update the Intel graphics driver.
2. Create a Python 3.11 virtual environment.
3. Install the XPU build of PyTorch from the PyTorch XPU wheel index (`pip install torch --index-url https://download.pytorch.org/whl/xpu`) — check the current PyTorch "Getting Started on Intel GPU" page for the exact command and supported Python versions.
4. Run the probe below. Expect `xpu: True`.
5. Install the rest of the dependencies from `requirements.txt`.

### 3.2 XPU probe (`scripts/probe_xpu.py`)

```python
import time, torch
print(torch.__version__, "xpu:", torch.xpu.is_available())
d = torch.device("xpu")

def bench(fn, n=20):
    for _ in range(3): fn()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n

N = 4096
for dt in (torch.float32, torch.float16):
    a = torch.randn(N, N, device=d, dtype=dt); b = torch.randn(N, N, device=d, dtype=dt)
    print(dt, f"{2*N**3 / bench(lambda: a @ b) / 1e12:.2f} TFLOPS")

x = torch.empty(64 * 2**20, device=d)  # 256 MB of fp32
print(f"copy bandwidth: {2 * x.numel() * 4 / bench(lambda: x.clone()) / 1e9:.1f} GB/s")
```

**How to read it:** expect a low single-digit TFLOPS number, well below a T4. If fp16 is *much* faster than fp32, the GPU is using matrix engines; if it is about the same or ~2×, it isn't — which only affects absolute iGPU numbers. The copy bandwidth is the real memory bandwidth you will use for the roofline plot (theoretical peak depends on whether your laptop has DDR5-5600 or LPDDR5x-7467).

### 3.3 Probe results (fill in)

| Metric | Value |
|---|---|
| fp32 TFLOPS | _TBD_ |
| fp16 TFLOPS | _TBD_ |
| Copy bandwidth (GB/s) | _TBD_ |
| RAM type / speed | _TBD_ |
| Driver version | _TBD_ |
| PyTorch version | _TBD_ |

### 3.4 RAM discipline (16 GB shared with the iGPU)

- Windows typically lets the iGPU borrow up to about half of system RAM; the "128 MB" shown in Settings is only the reserved slice.
- `numpy.memmap` for all token shards; never `read()` the whole 2 GB corpus into one string.
- Train the tokenizer on a 100–200 MB subset; encode the full corpus in chunks.
- Close browsers and other heavy apps for local benchmark runs.
- Keep laptop batch sizes small; use gradient accumulation instead.

### 3.5 Benchmark hygiene on a laptop

- Plugged in, "Best performance" power mode.
- Cooldown between configs; watch for clock drops (Task Manager → Performance).
- Don't run other GPU work at the same time (browser video, etc.).
- Record the power mode in each result JSON.

## 4. Reproducibility

| Item | Practice |
|---|---|
| Dependencies | Pinned `requirements.txt` (separate `requirements-xpu.txt` and `requirements-cuda.txt` if wheels differ) |
| Seeds | One `set_seed()` covering `random`, `numpy`, `torch`, and device RNGs; seed stored in every checkpoint and result |
| Configs | Every run reads a YAML; the resolved config is saved next to outputs |
| Environment report | `scripts/env_report.py` writes device, driver, versions, git commit → embedded in every result JSON |
| Determinism | `torch.use_deterministic_algorithms` for correctness tests only; benchmarks favor speed |
| Data versioning | Record token counts and SHA-256 of `train.bin` / `val.bin` in the README |

## 5. Directory conventions

```
data/raw/        # downloaded text (gitignored)
data/tok/        # tokenizer.json
data/shards/     # train.bin, val.bin (gitignored)
runs/<name>/     # ckpt.pt, config.yaml, log.csv (gitignored except log.csv)
results/         # benchmark JSON (committed)
results/samples/ # saved generations (committed)
```

## 6. Known issues log

| Date | Machine | Issue | Workaround / status |
|---|---|---|---|
| | | | |
