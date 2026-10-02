"""Tiny LR sweep on a small config to pick the peak learning rate before the big run.

    python scripts/lr_sweep.py --config configs/target_28m.yaml --steps 300 --lrs 3e-4,6e-4,1e-3,2e-3,3e-3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.train import load_config, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default="3e-4,6e-4,1e-3,2e-3,3e-3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    results = []
    for lr in [float(x) for x in args.lrs.split(",")]:
        cfg = load_config(args.config, args.set)
        t = cfg["train"]
        t.update(lr=lr, max_steps=args.steps, warmup_steps=max(10, args.steps // 10), eval_interval=args.steps,
                 ckpt_interval_steps=0, ckpt_interval_minutes=0, out_dir=f"runs/lr_sweep/lr_{lr:g}",
                 eval_batches=20)
        try:
            s = train(cfg, device_name=args.device)
            results.append((lr, s["final_val_loss"]))
        except RuntimeError as e:
            results.append((lr, float("nan")))
            print(f"lr={lr:g} diverged: {e}")
    print("\n  lr        val loss")
    for lr, v in results:
        print(f"  {lr:<9g} {v:.4f}")
    ok = [r for r in results if r[1] == r[1]]
    if ok:
        print(f"\nbest: lr={min(ok, key=lambda r: r[1])[0]:g}  (pick slightly below it for the long run)")


if __name__ == "__main__":
    main()
