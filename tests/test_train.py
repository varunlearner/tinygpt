import copy
import math

import torch

from tinygpt.train import get_lr, load_config, make_optimizer, train
from tests.conftest import make_model

ROOT_CFG = "configs/tiny_smoke.yaml"


def _cfg(toy_dir, out, **train_kw):
    cfg = load_config(ROOT_CFG)
    cfg["model"].update(vocab_size=400)
    cfg["train"].update(data_dir=str(toy_dir / "shards"), out_dir=str(out), tokenizer=str(toy_dir / "tok.json"),
                        micro_batch_size=8, grad_accum_steps=2, max_steps=20, eval_interval=10, eval_batches=2,
                        log_interval=5, ckpt_interval_steps=0, ckpt_interval_minutes=0, **train_kw)
    return cfg


def test_lr_schedule_shape():
    lr, mn, w, T = 1e-3, 1e-4, 10, 100
    assert get_lr(0, T, lr, mn, w) < lr * 0.2                         # warmup starts low
    assert abs(get_lr(w - 1, T, lr, mn, w) - lr) < 1e-9               # reaches peak
    assert get_lr(50, T, lr, mn, w) < lr                              # decays
    assert abs(get_lr(T, T, lr, mn, w) - mn) < 1e-12                  # ends at min
    assert all(get_lr(s, T, lr, mn, w) >= mn - 1e-12 for s in range(T + 1))


def test_optimizer_param_groups():
    m = make_model()
    opt = make_optimizer(m, 1e-3, 0.1, [0.9, 0.95], torch.device("cpu"))
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(m.tok_emb.weight) not in decay_ids                       # embedding not decayed
    assert id(m.blocks[0].attn_norm.weight) not in decay_ids           # norms not decayed
    assert id(m.blocks[0].attn.wq.weight) in decay_ids                 # matrices decayed
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_training_reduces_loss(toy_dir, tmp_path):
    cfg = _cfg(toy_dir, tmp_path / "run")
    cfg["train"].update(max_steps=60, lr=3e-3, warmup_steps=5, eval_interval=30)
    summary = train(cfg)
    assert summary["status"] == "finished"
    assert summary["final_val_loss"] < math.log(400) * 0.7             # far below the uniform-guess loss
    assert (tmp_path / "run" / "model.pt").exists() and (tmp_path / "run" / "train_log.csv").exists()


def test_exact_resume(toy_dir, tmp_path):
    """T14: 20 straight steps == 10 steps + kill + resume + 10 steps (bit-for-bit on CPU)."""
    a = train(_cfg(toy_dir, tmp_path / "straight"))
    cfg_b = _cfg(toy_dir, tmp_path / "resumed", stop_after_steps=10)
    first = train(cfg_b)
    assert first["status"] == "paused" and first["steps"] == 10
    cfg_b2 = copy.deepcopy(cfg_b)
    cfg_b2["train"]["stop_after_steps"] = None
    second = train(cfg_b2, resume=True)
    assert second["status"] == "finished" and second["steps"] == 20
    sd_a = torch.load(tmp_path / "straight" / "model.pt", weights_only=False)["model"]
    sd_b = torch.load(tmp_path / "resumed" / "model.pt", weights_only=False)["model"]
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    assert abs(a["final_val_loss"] - second["final_val_loss"]) < 1e-9


def test_distillation_runs(toy_dir, tmp_path):
    teacher_cfg = _cfg(toy_dir, tmp_path / "teacher")
    train(teacher_cfg)
    cfg = _cfg(toy_dir, tmp_path / "student", distill_teacher=str(tmp_path / "teacher" / "model.pt"), distill_alpha=0.3)
    cfg["model"].update(d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, mlp_hidden=64)
    summary = train(cfg)
    assert summary["status"] == "finished" and math.isfinite(summary["final_val_loss"])
