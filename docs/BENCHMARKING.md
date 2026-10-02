# Benchmarking and Testing Plan

**Last updated:** 2026-09-30 · **Status:** Harness implemented (`bench/`), correctness tests implemented (`tests/`); **no results yet**. All "expected" statements are hypotheses to be tested, not claims. Commands: [`RUNBOOK.md`](RUNBOOK.md).

## 1. Rules of the game

1. **No benchmark before its correctness test passes** (§3).
2. **Every number has an experiment ID and a committed JSON** in `results/`.
3. **Write your prediction first** (in `LEARNINGS.md`), then run, then explain the gap.
4. **Report the boring and the bad.** A speedup <1× is a result.
5. **Same prompts, same seeds, same generated length** across every compared method.

## 2. Methodology

| Topic | Rule |
|---|---|
| Warmup | ≥3 runs discarded (first XPU run can spend minutes compiling kernels; first CUDA run pays cuBLAS/cuDNN init) |
| Timing | `perf_counter` around the region **after** a device sync; sync again before stopping the clock (`torch.cuda.synchronize` / `torch.xpu.synchronize`) |
| Repeats | ≥10 timed runs; report **median** and **IQR** (p25–p75) |
| Prompts | Fixed set of 100 prompts from the validation set, saved in `bench/prompts.json`; fixed prompt length for batch experiments |
| Length control | Suppress EOS during throughput runs so every run generates exactly N tokens |
| Precision | State dtype in every result (fp32 for correctness; fp16 for speed on GPUs; fp32/bf16 on CPU as supported) |
| Metrics | decode tokens/s, time-to-first-token (prefill latency), end-to-end latency, peak memory, acceptance rate α, mean accepted tokens per round, speedup vs baseline |
| Environment capture | Device name, driver, torch version, dtype, Python version, git commit, power mode, run timestamp — stored in each JSON |
| Laptop hygiene | Plugged in, "Best performance" power mode, other apps closed, cooldown 30–60 s between configs, note if throttling suspected |
| Kaggle hygiene | Record `nvidia-smi` output; GPU model can differ between sessions — never mix sessions in one plot without saying so |
| Variance | If IQR > 10% of the median, investigate (background load? clocks?) before trusting the number |

**Memory-bandwidth utilization (MBU)** = `(bytes read per token × tokens/s) / peak_bandwidth`, where bytes per token = weight bytes + KV bytes read at that position. Peak bandwidth comes from the measured copy benchmark in `scripts/probe_xpu.py` (iGPU) or the vendor spec (T4).

## 3. Correctness tests (must pass before any benchmark)

| ID | Test | Pass criterion |
|---|---|---|
| T1 | Tokenizer round-trip | `decode(encode(x)) == x` for 100% of the val set |
| T2 | Tokenizer parity vs HF `tokenizers` | Bytes/token within 2%; ids differences understood |
| T3 | Model shapes + param count | Within 2% of target |
| T4 | Causality | Changing token at position *t* never changes logits at positions < *t* |
| T5 | RoPE offset | Logits at position *t* identical whether computed in a full pass or as a single token with `start_pos=t` |
| T6 | **KV-cache equivalence** | Prefill + step-by-step decode logits match full forward: max abs diff < 1e-4 (fp32); greedy tokens identical on 1000 prompts |
| T7 | Cache truncate | After `truncate(n)` and re-decode, logits equal the original path |
| T8 | Batched decode | Each row matches its single-sequence result (equal-length prompts) |
| T9 | **Speculative greedy identity** | Speculative output tokens == target-only greedy output, for k ∈ {1…8} and 1000 prompts (fp32) |
| T10 | **Speculative distribution test** | Toy target/draft (vocab 8, random weights): over ≥200k samples, chi-square of first-token *and* 2-token joint distribution vs the exact target distribution gives p > 0.01 (repeat with several seeds) |
| T11 | Speculative edge cases | k=1; EOS inside the draft block; generation ending at `max_len`; all-k-accepted; first-token rejection |
| T12 | Export parity | HF Llama / OpenVINO logits vs native: max abs diff < 1e-3 on fixed inputs |
| T13 | Quantization quality | Val-loss delta reported; generation samples inspected |
| T14 | Training resume | Loss curve continues smoothly across a save/kill/resume |

Where they live: T1 `test_tokenizer.py` · T2 `scripts/tokenizer_parity.py` + `test_export.py` · T3-T5 `test_model.py` ·
T6-T8 `test_kv_cache.py` · T9-T11 `test_speculative.py` · T12-T13 `test_export.py` · T14 `test_train.py`. Run all with `python -m pytest`.

Note on fp16: greedy identity can legitimately break on near-ties from rounding differences. Prove exactness in fp32; benchmark in fp16 and check the output agreement rate separately.

## 4. Experiment matrix

| ID | Question | Variables | Metrics | Devices | Pri |
|---|---|---|---|---|---|
| **E1** | How does KV cache speedup scale with generation length? | New tokens ∈ {32, 64, 128, 256, 480} (prompt 16 + 480 fits the 512 context); batch 1; naive vs cached | tokens/s, latency, speedup | T4, iGPU, CPU | P0 |
| **E2** | How does throughput scale with batch size? | Batch ∈ {1, 2, 4, 8, 16, 32, 64}; cached | tokens/s, peak memory | T4, iGPU, CPU | P0 |
| **E3** | When does speculative decoding help? | k ∈ {1,2,3,4,5,6,8}; temperature ∈ {0, 0.7, 1.0}; draft ∈ {2L-d256, 4L-d256, 4L-d256-distilled} | speedup vs cached baseline, α, mean accepted/round | T4, iGPU | P0 |
| **E4** | Does theory predict the measured speedup? | Fit α and c per config; plug into formula | predicted vs measured speedup, residual | T4, iGPU | P1 |
| **E5** | How much is launch overhead? | Decode-step time, eager vs `torch.compile` (default) vs `reduce-overhead` (CUDA graphs, CUDA only) | step time, `c` before/after | T4 (+CPU for compile) | P1 |
| **E6** | How close are we to the memory-bandwidth roofline? | Batch ∈ {1, 4, 16}; fp32 vs fp16 | MBU, tokens/s vs bandwidth ceiling | iGPU, T4 | P1 |
| **E7** | What does int8 weight-only cost/gain? | fp16 vs int8 | tokens/s, val-loss delta | T4, iGPU, CPU | P2 |
| **E8** | How does our engine compare with others? | Native vs HF `generate` vs OpenVINO vs llama.cpp | tokens/s at batch 1 | CPU, iGPU | P2 |

### Hypotheses to record before running (fill in `LEARNINGS.md`)

- **H1.** Naive per-token cost grows with length; cached cost is roughly flat → speedup rises with length.
- **H2.** At batch 1 with a model this small, decode is launch-overhead-bound, not bandwidth-bound → tokens/s far below the roofline ceiling.
- **H3.** Speculative decoding speedup is limited by `c` (draft cost ratio), which eager-mode overhead pushes toward 1; CUDA graphs should lower it.
- **H4.** TinyStories text is highly predictable → high acceptance rates (α) at low temperature, dropping as temperature rises.
- **H5.** A distilled draft beats an independently trained one at equal size.
- **H6.** Batching helps throughput a lot on the iGPU/CPU at small batch (bandwidth-starved), flattening as compute saturates.

## 5. Analysis to write up (per experiment)

1. Plot (generated by script; axes labeled with units; hardware in the title).
2. One paragraph: what happened.
3. One paragraph: why (cite profiler data or the roofline where possible).
4. One paragraph: how it compared with the pre-registered prediction.
5. Caveats: dtype, eager vs compiled, thermal notes, number of runs, IQR.

Use `torch.profiler` on the decode step to back up "launch-bound" claims: many tiny kernels with GPU idle gaps is the signature.

## 6. Result JSON schema (per run file, written by `bench.run`)

```json
{
  "experiment": "E3",
  "env": {"device": "cuda", "device_name": "Tesla T4", "dtype": "fp16", "torch": "...", "python": "...",
          "git_commit": "...", "power_mode": "...", "copy_bandwidth_gbs": 0.0, "timestamp": "..."},
  "config": {"...": "all CLI arguments of the run"},
  "rows": [
    {"kind": "speculative", "draft": "draft_4l", "k": 4, "temperature": 0.0, "tokens_per_s": 0.0,
     "seconds": {"median": 0.0, "p25": 0.0, "p75": 0.0, "n": 10}, "speedup_median": 0.0,
     "acceptance_rate": 0.0, "mean_tokens_per_round": 0.0, "c": 0.0}
  ]
}
```

Files land in `results/` as `<experiment>_<device>_<timestamp>.json`; `bench.analyze` adds `analysis_E4_*` and `analysis_E6_*`;
`bench.plot` renders `results/plots/*.png` and `results/SUMMARY.md`.

## 7. Results (fill in as experiments complete)

### E1 — Naive vs KV cache (batch 1, fp16, T4)

| New tokens | Naive tok/s | Cached tok/s | Speedup |
|---|---|---|---|
| 32 | _TBD_ | _TBD_ | _TBD_ |
| 64 | _TBD_ | _TBD_ | _TBD_ |
| 128 | _TBD_ | _TBD_ | _TBD_ |
| 256 | _TBD_ | _TBD_ | _TBD_ |
| 480 | _TBD_ | _TBD_ | _TBD_ |

### E3 — Speculative decoding (T4, fp16, temperature 0)

| Draft | k | α | Mean accepted / round | Tok/s | Speedup vs cached |
|---|---|---|---|---|---|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### E4 — Theory vs measured

| Config | α | c | Predicted speedup | Measured speedup | Residual |
|---|---|---|---|---|---|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Cross-device summary (batch 1, cached, fp16 where supported)

| Device | Execution mode | Tok/s | MBU |
|---|---|---|---|
| T4 | eager | _TBD_ | _TBD_ |
| T4 | compiled / CUDA graphs | _TBD_ | _TBD_ |
| Arc iGPU (155H) | XPU eager | _TBD_ | _TBD_ |
| Arc iGPU (155H) | OpenVINO | _TBD_ | _TBD_ |
| CPU (155H) | PyTorch | _TBD_ | _TBD_ |
| CPU (155H) | OpenVINO | _TBD_ | _TBD_ |
