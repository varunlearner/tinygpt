"""Training loop: fp16/bf16 autocast, gradient accumulation, AdamW, warmup + cosine LR, gradient clipping,
exact checkpoint/resume, optional distillation from a teacher, optional time limit (for Kaggle sessions).

    python -m tinygpt.train --config configs/target_28m.yaml
    python -m tinygpt.train --config configs/target_28m.yaml --resume          # continue after a crash / session end
    python -m tinygpt.train --config configs/draft_4m.yaml --set train.distill_teacher=runs/target_28m/model.pt

Any config value can be overridden:  --set train.lr=5e-4 --set model.n_layers=6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from tinygpt.data.loader import TokenData
from tinygpt.model.config import GPTConfig
from tinygpt.model.gpt import GPT
from tinygpt.model.io import load_model, save_model
from tinygpt.utils import best_train_dtype, env_report, make_generator, resolve_device, set_seed, sync, torch_dtype

TRAIN_DEFAULTS: Dict[str, Any] = dict(
    data_dir="data/shards", out_dir="runs/run", tokenizer="data/tok/tokenizer.json", seed=1337,
    micro_batch_size=32, grad_accum_steps=4, epochs=1.0, max_steps=None,
    lr=1e-3, min_lr_ratio=0.1, warmup_steps=500, weight_decay=0.1, betas=[0.9, 0.95], grad_clip=1.0,
    dtype="auto", compile=False,
    eval_interval=500, eval_batches=50, log_interval=20,
    ckpt_interval_steps=500, ckpt_interval_minutes=15.0, time_limit_minutes=None, stop_after_steps=None,
    sample_prompt="Once upon a time",
    distill_teacher=None, distill_alpha=0.5,
)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------

def load_config(path: str, overrides: Optional[list] = None) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    cfg["train"] = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    for item in overrides or []:
        key, _, val = item.partition("=")
        section, _, name = key.partition(".")
        if section not in ("model", "train") or not name:
            raise ValueError(f"--set expects model.<key>=<v> or train.<key>=<v>, got {item!r}")
        cfg[section][name] = yaml.safe_load(val)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return cfg


def get_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def make_optimizer(model: GPT, lr: float, weight_decay: float, betas, device: torch.device):
    """Weight decay only on 2-D weight matrices (not norms, not the embedding table)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and not name.startswith("tok_emb") else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = dict(lr=lr, betas=tuple(betas))
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def make_scaler(device_type: str, enabled: bool):
    if not enabled:
        return torch.amp.GradScaler(device_type if device_type in ("cuda", "cpu") else "cuda", enabled=False)
    try:
        return torch.amp.GradScaler(device_type, enabled=True)
    except Exception:  # very old torch
        return torch.cuda.amp.GradScaler(enabled=True)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, teacher_logits: Optional[torch.Tensor], alpha: float):
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1))
    if teacher_logits is None:
        return ce, ce.detach()
    kl = F.kl_div(F.log_softmax(logits.float().view(-1, V), dim=-1),
                  F.softmax(teacher_logits.float().view(-1, V), dim=-1), reduction="batchmean")
    return alpha * ce + (1.0 - alpha) * kl, ce.detach()


@torch.no_grad()
def estimate_val_loss(model, data: TokenData, tc: dict, ctx: int, device, autocast_ctx) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for x, y in data.val_batches(tc["eval_batches"], tc["micro_batch_size"], ctx, device):
        with autocast_ctx():
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train(was_training)
    return float(np.mean(losses))


def _append_csv(path: str, header: list, row: list) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# ------------------------------------------------------------------------------------------------
# main training function
# ------------------------------------------------------------------------------------------------

def train(cfg: Dict[str, Any], resume: bool = False, device_name: str = "auto") -> Dict[str, Any]:
    tc = cfg["train"]
    device = resolve_device(device_name)
    dtype_name = best_train_dtype(device) if tc["dtype"] == "auto" else tc["dtype"]
    amp_dtype = torch_dtype(dtype_name)
    use_amp = dtype_name != "fp32"
    autocast_ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = make_scaler(device.type, enabled=(dtype_name == "fp16" and device.type in ("cuda", "xpu")))

    os.makedirs(tc["out_dir"], exist_ok=True)
    set_seed(tc["seed"])
    data = TokenData(tc["data_dir"])
    mcfg = GPTConfig.from_dict(cfg["model"])
    if mcfg.vocab_size < data.vocab_size:
        raise ValueError(f"model vocab_size {mcfg.vocab_size} < tokenizer vocab {data.vocab_size}")
    if mcfg.vocab_size != data.vocab_size:
        print(f"[train] note: model vocab_size={mcfg.vocab_size} > data vocab {data.vocab_size} (fine, extra rows unused)")
    ctx = mcfg.max_seq_len

    model = GPT(mcfg).to(device)
    model.train()
    teacher = None
    if tc["distill_teacher"]:
        teacher = load_model(tc["distill_teacher"], device)
        teacher.requires_grad_(False)
        assert teacher.cfg.vocab_size == mcfg.vocab_size, "teacher/student vocab mismatch"
        print(f"[train] distilling from {tc['distill_teacher']} (alpha={tc['distill_alpha']})")

    opt = make_optimizer(model, tc["lr"], tc["weight_decay"], tc["betas"], device)
    fwd_model = torch.compile(model) if tc["compile"] else model

    tokens_per_step = tc["micro_batch_size"] * tc["grad_accum_steps"] * ctx
    max_steps = tc["max_steps"] or int(math.ceil(tc["epochs"] * data.meta["train_tokens"] / tokens_per_step))
    min_lr = tc["lr"] * tc["min_lr_ratio"]
    rng = np.random.default_rng(tc["seed"])

    step, tokens_seen, best_val = 0, 0, float("inf")
    latest = os.path.join(tc["out_dir"], "ckpt_latest.pt")
    if resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, tokens_seen, best_val = ck["step"], ck["tokens_seen"], ck["best_val"]
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        rng.bit_generator.state = ck["rng"]["numpy_data"]
        if ck.get("max_steps") not in (None, max_steps):
            print(f"[train] WARNING: max_steps changed {ck['max_steps']} -> {max_steps}; LR schedule will differ")
        print(f"[train] resumed from {latest} at step {step} ({tokens_seen / 1e6:.1f}M tokens)")
    elif resume:
        print("[train] --resume given but no checkpoint found; starting fresh")

    n_params = model.num_params()
    print(f"[train] {cfg['name']}: {n_params / 1e6:.2f}M params | device={device} dtype={dtype_name} | "
          f"{tokens_per_step:,} tokens/step | {max_steps:,} steps ({max_steps * tokens_per_step / 1e6:,.0f}M tokens)")
    with open(os.path.join(tc["out_dir"], "config.resolved.json"), "w") as f:
        json.dump({**cfg, "resolved": {"dtype": dtype_name, "max_steps": max_steps, "params": n_params,
                                        "tokens_per_step": tokens_per_step}, "env": env_report(device, dtype_name)}, f, indent=2)

    def save_ckpt() -> None:
        obj = {
            "model_config": mcfg.to_dict(), "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step, "tokens_seen": tokens_seen, "best_val": best_val, "max_steps": max_steps,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "numpy_data": rng.bit_generator.state},
            "train_cfg": tc,
        }
        tmp = latest + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, latest)                      # atomic: a crash never leaves a half-written checkpoint

    log_path = os.path.join(tc["out_dir"], "train_log.csv")
    eval_path = os.path.join(tc["out_dir"], "eval_log.csv")
    t_start = time.time()
    t_last_ckpt = t_start
    t_window, tok_window, loss_window = time.time(), 0, []
    paused = False

    while step < max_steps:
        lr = get_lr(step, max_steps, tc["lr"], min_lr, tc["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(tc["grad_accum_steps"]):
            x, y = data.get_batch("train", tc["micro_batch_size"], ctx, rng, device)
            with autocast_ctx():
                logits = fwd_model(x)
                t_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                loss, ce = compute_loss(logits, y, t_logits, tc["distill_alpha"])
            scaler.scale(loss / tc["grad_accum_steps"]).backward()
            step_loss += ce / tc["grad_accum_steps"]
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]) if tc["grad_clip"] else torch.tensor(0.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens_seen += tokens_per_step
        tok_window += tokens_per_step
        loss_window.append(step_loss)

        if step % tc["log_interval"] == 0 or step == max_steps:
            mean_loss = float(torch.stack(loss_window).mean())          # one device sync per log interval
            sync(device)
            now = time.time()
            tps = tok_window / max(now - t_window, 1e-9)
            print(f"step {step:6d}/{max_steps} | loss {mean_loss:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f} | "
                  f"{tps / 1e3:6.1f}k tok/s | {(now - t_start) / 60:6.1f} min")
            _append_csv(log_path, ["step", "tokens", "loss", "lr", "grad_norm", "tokens_per_s", "elapsed_min"],
                        [step, tokens_seen, f"{mean_loss:.5f}", f"{lr:.6e}", f"{float(gnorm):.4f}", f"{tps:.0f}",
                         f"{(now - t_start) / 60:.2f}"])
            if not math.isfinite(mean_loss):
                raise RuntimeError("loss is NaN/inf - lower the LR, check dtype, or resume from an earlier checkpoint")
            t_window, tok_window, loss_window = now, 0, []

        if step % tc["eval_interval"] == 0 or step == max_steps:
            val = estimate_val_loss(model, data, tc, ctx, device, autocast_ctx)
            print(f"  >> eval @ step {step}: val loss {val:.4f} (ppl {math.exp(val):.2f})")
            _append_csv(eval_path, ["step", "tokens", "val_loss"], [step, tokens_seen, f"{val:.5f}"])
            if val < best_val:
                best_val = val
                save_model(model, os.path.join(tc["out_dir"], "model_best.pt"), {"val_loss": val, "step": step})
            t_window = time.time()

        now = time.time()
        due_steps = tc["ckpt_interval_steps"] and step % tc["ckpt_interval_steps"] == 0
        due_time = tc["ckpt_interval_minutes"] and (now - t_last_ckpt) > tc["ckpt_interval_minutes"] * 60
        if due_steps or due_time:
            save_ckpt()
            t_last_ckpt = now
        if tc["time_limit_minutes"] and (now - t_start) > tc["time_limit_minutes"] * 60 and step < max_steps:
            save_ckpt()
            print(f"[train] time limit reached at step {step}. Checkpoint saved. Re-run with --resume to continue.")
            paused = True
            break
        if tc["stop_after_steps"] and step >= tc["stop_after_steps"] and step < max_steps:
            save_ckpt()
            print(f"[train] stop_after_steps={tc['stop_after_steps']} reached. Checkpoint saved (resume with --resume).")
            paused = True
            break

    summary = {"name": cfg["name"], "status": "paused" if paused else "finished", "steps": step,
               "tokens_seen": tokens_seen, "params": n_params, "best_val_loss": best_val,
               "minutes": (time.time() - t_start) / 60, "dtype": dtype_name, "device": str(device)}
    if not paused:
        val = estimate_val_loss(model, data, tc, ctx, device, autocast_ctx)
        summary["final_val_loss"] = val
        save_model(model, os.path.join(tc["out_dir"], "model.pt"),
                   {"val_loss": val, "step": step, "tokens_seen": tokens_seen})
        save_ckpt()
        try:
            _write_sample(model, tc, device, os.path.join(tc["out_dir"], "sample.txt"))
        except Exception as e:  # a cosmetic step must never kill a finished run
            print(f"[train] (sample generation skipped: {type(e).__name__}: {e})")
        print(f"[train] done. final val loss {val:.4f} -> {tc['out_dir']}/model.pt")
    with open(os.path.join(tc["out_dir"], "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def _write_sample(model: GPT, tc: dict, device, path: str) -> None:
    """Generate one short sample after training (skipped if the tokenizer file is missing)."""
    if not os.path.exists(tc["tokenizer"]):
        return
    from tinygpt.infer.generate import generate_cached
    from tinygpt.infer.sampling import SamplingConfig
    from tinygpt.tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(tc["tokenizer"])
    model.eval()
    ids = torch.tensor([tok.encode(tc["sample_prompt"])], device=device)
    n_new = min(200, model.cfg.max_seq_len - ids.shape[1])
    out = generate_cached(model, ids, n_new, SamplingConfig(temperature=0.8, top_k=40), eos_id=tok.eos_id,
                          generator=make_generator(device, 0))
    text = tok.decode(out[0].tolist())
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("[train] sample:\n" + text)
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="use torch.compile (recommended on CUDA)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--time-limit-minutes", type=float, default=None,
                    help="stop gracefully (checkpoint saved) after this many minutes; resume later")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=5e-4 (repeatable)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.compile:
        cfg["train"]["compile"] = True
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.time_limit_minutes is not None:
        cfg["train"]["time_limit_minutes"] = args.time_limit_minutes
    train(cfg, resume=args.resume, device_name=args.device)


if __name__ == "__main__":
    main()
