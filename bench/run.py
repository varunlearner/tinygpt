"""Benchmark harness.

    python -m bench.run E1 --target runs/target_28m/model.pt --data-dir data/shards --device cuda
    python -m bench.run E2 --target ...
    python -m bench.run E3 --target ... --drafts draft_4l=runs/draft_4m/model.pt,draft_4l_distilled=runs/draft_4m_distilled/model.pt
    python -m bench.run E5 --target ... --draft runs/draft_4m/model.pt        # launch-overhead / torch.compile study

Experiments (see docs/BENCHMARKING.md):
    E1  naive vs KV-cache vs generation length          E2  throughput vs batch size
    E3  speculative decoding: k x temperature x draft   E5  eager vs torch.compile decode-step time
E4 (theory vs measured) and E6 (roofline / MBU) are computed from these results by `python -m bench.analyze`.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

import torch

from bench.common import get_prompts, iqr_warning, save_result, summarize, timed
from tinygpt.infer.generate import generate_cached, generate_naive
from tinygpt.infer.sampling import SamplingConfig
from tinygpt.infer.speculative import SpecStats, generate_speculative
from tinygpt.model.io import load_model
from tinygpt.model.kv_cache import KVCache
from tinygpt.utils import (env_report, make_generator, peak_memory_bytes, reset_peak_memory, resolve_device, set_seed, sync,
                           torch_dtype)

GREEDY = SamplingConfig(temperature=0.0)


def csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]


def csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x]


def make_gen(device, seed: int):
    return make_generator(device, seed)


# ------------------------------------------------------------------------------------------------
# measuring the per-step cost of a model (used for the cost coefficient c)
# ------------------------------------------------------------------------------------------------

@torch.no_grad()
def measure_decode_step(model, device, ctx_len: int = 64, steps: int = 60, warmup: int = 10,
                        compile_mode: str | None = None) -> float:
    """Median seconds for one T=1 cached decode step at a fixed cache length (rolled back every step)."""
    fwd = model
    if compile_mode:
        fwd = torch.compile(model, mode=None if compile_mode == "default" else compile_mode)
    cache = KVCache.for_model(model, 1)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, ctx_len), device=device)
    model(prompt, cache=cache, logits_last=1)
    tok = prompt[:, -1:]
    times = []
    for i in range(warmup + steps):
        cache.truncate(ctx_len)
        sync(device)
        t0 = time.perf_counter()
        fwd(tok, cache=cache, logits_last=1)
        sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ------------------------------------------------------------------------------------------------
# E1: naive vs KV cache
# ------------------------------------------------------------------------------------------------

def run_e1(args, target, device, env) -> None:
    prompts = get_prompts(args.data_dir, 1, args.prompt_len)
    prompt = torch.tensor([prompts[0]], device=device)
    rows = []
    for L in args.lengths:
        if args.prompt_len + L > target.cfg.max_seq_len:
            print(f"[E1] skipping L={L}: exceeds context")
            continue
        row = {"new_tokens": L, "prompt_len": args.prompt_len, "batch": 1}
        for method, fn in (("naive", generate_naive), ("cached", generate_cached)):
            if method == "naive" and L > args.naive_max:
                continue
            runs = args.runs if method == "cached" else max(3, min(args.runs, args.naive_runs))
            times = timed(lambda: fn(target, prompt, L, GREEDY), args.warmup, runs, device)
            s = summarize(times)
            row[method] = {"seconds": s, "tokens_per_s": L / s["median"]}
            print(f"[E1] L={L:4d} {method:6s}: {s['median'] * 1e3:9.1f} ms  {L / s['median']:9.1f} tok/s"
                  f"{iqr_warning(s, method)}")
        if "naive" in row:
            row["speedup"] = row["naive"]["seconds"]["median"] / row["cached"]["seconds"]["median"]
            print(f"[E1] L={L:4d} speedup cached/naive = {row['speedup']:.2f}x")
        rows.append(row)

    # time-to-first-token (prefill latency)
    cache = KVCache.for_model(target, 1)
    ttft = summarize(timed(lambda: (cache.reset(), target(prompt, cache=cache, logits_last=1)), args.warmup, args.runs, device))
    save_result("E1", env, vars_of(args), rows, args.out, {"ttft_seconds": ttft, "params": target.num_params(),
                                                         "model_config": target.cfg.to_dict()})


# ------------------------------------------------------------------------------------------------
# E2: batch scaling
# ------------------------------------------------------------------------------------------------

def run_e2(args, target, device, env) -> None:
    prompts = get_prompts(args.data_dir, max(args.batches), args.prompt_len)
    rows = []
    for B in args.batches:
        prompt = torch.tensor(prompts[:B], device=device)
        try:
            reset_peak_memory(device)
            cache = KVCache.for_model(target, B)
            times = timed(lambda: generate_cached(target, prompt, args.new_tokens, GREEDY, cache=cache),
                          args.warmup, args.runs, device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:   # out of memory at large batch
            print(f"[E2] B={B}: stopped ({type(e).__name__}: {str(e)[:80]})")
            break
        s = summarize(times)
        tps = B * args.new_tokens / s["median"]
        rows.append({"batch": B, "new_tokens": args.new_tokens, "prompt_len": args.prompt_len, "seconds": s,
                     "tokens_per_s": tps, "steps_per_s": args.new_tokens / s["median"],
                     "peak_memory_bytes": peak_memory_bytes(device), "kv_cache_bytes": cache.nbytes})
        print(f"[E2] B={B:3d}: {tps:10.1f} tok/s total  ({tps / B:8.1f} tok/s/seq){iqr_warning(s, f'B={B}')}")
    save_result("E2", env, vars_of(args), rows, args.out, {"params": target.num_params(),
                                                         "model_config": target.cfg.to_dict()})


# ------------------------------------------------------------------------------------------------
# E3: speculative decoding
# ------------------------------------------------------------------------------------------------

def run_e3(args, target, drafts: Dict[str, torch.nn.Module], device, env) -> None:
    prompts = get_prompts(args.data_dir, args.runs, args.prompt_len)
    prompt_t = [torch.tensor([p], device=device) for p in prompts]
    rows = []
    t_step = measure_decode_step(target, device)
    c_by_draft = {name: measure_decode_step(d, device) / t_step for name, d in drafts.items()}
    print(f"[E3] target decode step {t_step * 1e3:.2f} ms; cost ratios c = "
          + ", ".join(f"{n}: {c:.2f}" for n, c in c_by_draft.items()))

    for temp in args.temps:
        sampling = SamplingConfig(temperature=temp)
        # baseline: cached decoding of the target on the SAME prompts / seeds
        base = []
        generate_cached(target, prompt_t[0], args.new_tokens, sampling, generator=make_gen(device, 0))
        for i, p in enumerate(prompt_t):
            sync(device)
            t0 = time.perf_counter()
            generate_cached(target, p, args.new_tokens, sampling, generator=make_gen(device, 1000 + i))
            sync(device)
            base.append(time.perf_counter() - t0)
        base_tps = args.new_tokens / statistics.median(base)
        print(f"[E3] temp={temp}: cached baseline {base_tps:.1f} tok/s")
        rows.append({"kind": "baseline_cached", "temperature": temp, "tokens_per_s": base_tps,
                     "seconds": summarize(base)})

        for dname, draft in drafts.items():
            for k in args.ks:
                for w in range(args.warmup):
                    generate_speculative(target, draft, prompt_t[0], args.new_tokens, k, sampling,
                                         generator=make_gen(device, w))
                times, agg = [], SpecStats()
                for i, p in enumerate(prompt_t):
                    sync(device)
                    t0 = time.perf_counter()
                    _, st = generate_speculative(target, draft, p, args.new_tokens, k, sampling,
                                                 generator=make_gen(device, 1000 + i))
                    sync(device)
                    times.append(time.perf_counter() - t0)
                    for f in ("rounds", "proposed", "accepted", "tested", "generated", "target_forwards", "draft_forwards"):
                        setattr(agg, f, getattr(agg, f) + getattr(st, f))
                speedups = [b / t for b, t in zip(base, times)]
                s = summarize(times)
                row = {"kind": "speculative", "draft": dname, "k": k, "temperature": temp,
                       "tokens_per_s": args.new_tokens / s["median"], "seconds": s,
                       "speedup_median": statistics.median(speedups),
                       "speedup_p25": sorted(speedups)[len(speedups) // 4],
                       "speedup_p75": sorted(speedups)[(3 * len(speedups)) // 4],
                       "acceptance_rate": agg.acceptance_rate, "mean_tokens_per_round": agg.mean_tokens_per_round,
                       "target_forwards_per_token": agg.target_forwards / max(agg.generated, 1),
                       "draft_forwards_per_token": agg.draft_forwards / max(agg.generated, 1),
                       "c": c_by_draft[dname]}
                rows.append(row)
                print(f"[E3] temp={temp:<4} {dname:22s} k={k}: alpha={row['acceptance_rate']:.2f} "
                      f"tok/round={row['mean_tokens_per_round']:.2f}  {row['tokens_per_s']:8.1f} tok/s  "
                      f"speedup x{row['speedup_median']:.2f}")
    save_result("E3", env, vars_of(args), rows, args.out,
                {"target_step_seconds": t_step, "c_by_draft": c_by_draft, "params": target.num_params(),
                 "draft_params": {n: d.num_params() for n, d in drafts.items()}})


# ------------------------------------------------------------------------------------------------
# E5: launch overhead (eager vs torch.compile)
# ------------------------------------------------------------------------------------------------

def run_e5(args, target, draft, device, env) -> None:
    rows = []
    for mode in (None, "default", "reduce-overhead"):
        label = "eager" if mode is None else f"compile:{mode}"
        if mode == "reduce-overhead" and device.type != "cuda":
            print(f"[E5] {label}: skipped (CUDA graphs need a CUDA device)")
            continue
        try:
            t_t = measure_decode_step(target, device, compile_mode=mode)
            t_d = measure_decode_step(draft, device, compile_mode=mode)
            rows.append({"mode": label, "target_step_ms": t_t * 1e3, "draft_step_ms": t_d * 1e3, "c": t_d / t_t})
            print(f"[E5] {label:22s} target {t_t * 1e3:7.3f} ms | draft {t_d * 1e3:7.3f} ms | c = {t_d / t_t:.2f}")
        except Exception as e:  # experimental: never crash the whole run
            rows.append({"mode": label, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[E5] {label}: FAILED ({type(e).__name__}: {str(e)[:100]})")
    save_result("E5", env, vars_of(args), rows, args.out)


def vars_of(args) -> dict:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["E1", "E2", "E3", "E5"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default=None, help="single draft (E5)")
    ap.add_argument("--drafts", default=None, help="name=path,name=path (E3)")
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="fp32|fp16|bf16 (auto: fp16 on cuda/xpu, fp32 on cpu)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--power-mode", default="unspecified", help="label recorded in the result (e.g. plugged-in/best-performance)")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--prompt-len", type=int, default=16)
    # E1
    ap.add_argument("--lengths", type=csv_ints, default=[32, 64, 128, 256, 480])
    ap.add_argument("--naive-max", type=int, default=10_000, help="skip the naive baseline above this length")
    ap.add_argument("--naive-runs", type=int, default=5, help="timed runs for the (slow) naive baseline")
    # E2
    ap.add_argument("--batches", type=csv_ints, default=[1, 2, 4, 8, 16, 32, 64])
    # E2/E3
    ap.add_argument("--new-tokens", type=int, default=None)
    # E3
    ap.add_argument("--ks", type=csv_ints, default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--temps", type=csv_floats, default=[0.0, 0.7, 1.0])
    args = ap.parse_args()

    if args.quick:
        args.runs, args.warmup, args.naive_runs = 3, 1, 3
        args.lengths, args.batches, args.ks, args.temps = [16, 32], [1, 2, 4], [2, 4], [0.0, 1.0]
    if args.new_tokens is None:
        args.new_tokens = 128 if args.experiment == "E2" else (32 if args.quick else 256)

    device = resolve_device(args.device)
    dtype = args.dtype if args.dtype != "auto" else ("fp16" if device.type in ("cuda", "xpu") else "fp32")
    set_seed(0)
    target = load_model(args.target, device, torch_dtype(dtype))
    env = env_report(device, dtype, measure_bandwidth=True, power_mode=args.power_mode)
    print(f"[bench] {args.experiment} on {env['device_name']} ({device}) dtype={dtype} | "
          f"target {target.num_params() / 1e6:.1f}M params | measured copy bandwidth {env.get('copy_bandwidth_gbs')} GB/s")

    if args.experiment == "E1":
        run_e1(args, target, device, env)
    elif args.experiment == "E2":
        run_e2(args, target, device, env)
    elif args.experiment == "E3":
        if not args.drafts:
            raise SystemExit("E3 needs --drafts name=path,name=path")
        drafts = {}
        for item in args.drafts.split(","):
            name, _, path = item.partition("=")
            drafts[name] = load_model(path, device, torch_dtype(dtype))
        run_e3(args, target, drafts, device, env)
    elif args.experiment == "E5":
        if not args.draft:
            raise SystemExit("E5 needs --draft")
        run_e5(args, target, load_model(args.draft, device, torch_dtype(dtype)), device, env)


if __name__ == "__main__":
    main()
