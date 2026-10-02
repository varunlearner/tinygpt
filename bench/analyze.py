"""Post-hoc analysis of benchmark results.

    python -m bench.analyze --results results

E4  theory vs measured speculative speedup:   S_theory = (1 - a^(k+1)) / ((1 - a) (k c + 1))
    a = measured acceptance rate, c = measured draft/target step-cost ratio, k = draft tokens per round.
E6  memory-bandwidth utilization (MBU) of cached decoding, from E1 (batch 1) and E2 (batch scaling):
    MBU = bytes_read_per_step * steps_per_s / measured_copy_bandwidth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

from tinygpt.model.config import GPTConfig

ELEM = {"fp32": 4, "fp16": 2, "bf16": 2}


def predicted_speedup(alpha: float, k: int, c: float) -> float:
    expected_tokens = (k + 1.0) if abs(1.0 - alpha) < 1e-9 else (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
    return expected_tokens / (k * c + 1.0)


def analyze_e4(path: str) -> dict:
    res = json.load(open(path))
    rows = []
    for r in res["rows"]:
        if r.get("kind") != "speculative":
            continue
        pred = predicted_speedup(r["acceptance_rate"], r["k"], r["c"])
        rows.append({"draft": r["draft"], "k": r["k"], "temperature": r["temperature"], "alpha": r["acceptance_rate"],
                     "c": r["c"], "predicted_speedup": pred, "measured_speedup": r["speedup_median"],
                     "residual_pct": 100.0 * (r["speedup_median"] / pred - 1.0)})
    return {"source": os.path.basename(path), "env": res["env"], "rows": rows}


def analyze_e6(path: str) -> dict:
    res = json.load(open(path))
    env, cfg = res["env"], GPTConfig.from_dict(res["model_config"])
    bw = (env.get("copy_bandwidth_gbs") or 0) * 1e9
    elem = ELEM[env["dtype"]]
    weight_bytes = res["params"] * elem
    rows = []
    for r in res["rows"]:
        if res["experiment"] == "E1":
            if "cached" not in r:
                continue
            batch, avg_len = 1, r["prompt_len"] + r["new_tokens"] / 2
            steps_per_s = r["cached"]["tokens_per_s"]
        else:
            batch, avg_len = r["batch"], r["prompt_len"] + r["new_tokens"] / 2
            steps_per_s = r["steps_per_s"]
        bytes_per_step = weight_bytes + batch * cfg.kv_bytes_per_token(elem) * avg_len
        rows.append({"batch": batch, "new_tokens": r["new_tokens"], "steps_per_s": steps_per_s,
                     "tokens_per_s": steps_per_s * batch, "bytes_per_step_MB": bytes_per_step / 1e6,
                     "mbu_pct": 100 * bytes_per_step * steps_per_s / bw if bw else None,
                     "bandwidth_ceiling_tokens_per_s": batch * bw / bytes_per_step if bw else None})
    return {"source": os.path.basename(path), "experiment": res["experiment"], "env": env, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(args.results, "E3_*.json"))):
        out = analyze_e4(path)
        dst = os.path.join(args.results, "analysis_E4_" + os.path.basename(path))
        json.dump(out, open(dst, "w"), indent=2)
        print(f"\n== E4 theory vs measured  ({out['env']['device_name']}) ==")
        print(f"{'draft':22s} {'temp':>5s} {'k':>2s} {'alpha':>6s} {'c':>5s} {'theory':>7s} {'measured':>8s} {'resid%':>7s}")
        for r in out["rows"]:
            print(f"{r['draft']:22s} {r['temperature']:5.1f} {r['k']:2d} {r['alpha']:6.2f} {r['c']:5.2f} "
                  f"{r['predicted_speedup']:7.2f} {r['measured_speedup']:8.2f} {r['residual_pct']:7.1f}")

    for pattern in ("E1_*.json", "E2_*.json"):
        for path in sorted(glob.glob(os.path.join(args.results, pattern))):
            out = analyze_e6(path)
            dst = os.path.join(args.results, "analysis_E6_" + os.path.basename(path))
            json.dump(out, open(dst, "w"), indent=2)
            print(f"\n== E6 roofline / MBU from {out['source']}  ({out['env']['device_name']}, "
                  f"measured copy bw {out['env'].get('copy_bandwidth_gbs')} GB/s) ==")
            print(f"{'batch':>5s} {'new':>5s} {'tok/s':>10s} {'MB/step':>9s} {'MBU%':>7s} {'bw-ceiling tok/s':>17s}")
            for r in out["rows"]:
                mbu = f"{r['mbu_pct']:.1f}" if r["mbu_pct"] is not None else "n/a"
                ceil = f"{r['bandwidth_ceiling_tokens_per_s']:.0f}" if r["bandwidth_ceiling_tokens_per_s"] else "n/a"
                print(f"{r['batch']:5d} {r['new_tokens']:5d} {r['tokens_per_s']:10.1f} {r['bytes_per_step_MB']:9.1f} {mbu:>7s} {ceil:>17s}")


if __name__ == "__main__":
    main()
