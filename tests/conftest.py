import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tinygpt.model.config import GPTConfig  # noqa: E402
from tinygpt.model.gpt import GPT  # noqa: E402


def make_model(seed: int = 0, vocab: int = 64, d: int = 64, layers: int = 3, heads: int = 4, kv: int = 4,
               hidden: int = 128, max_len: int = 128, std_scale: float = 1.0) -> GPT:
    """Small random model. std_scale > 1 makes the output distributions less trivial."""
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads, n_kv_heads=kv,
                    mlp_hidden=hidden, max_seq_len=max_len)
    m = GPT(cfg)
    if std_scale != 1.0:
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(std_scale)
    return m.eval()


@pytest.fixture
def small_model():
    return make_model(seed=0, std_scale=3.0)


@pytest.fixture
def gqa_model():
    return make_model(seed=1, kv=2, std_scale=3.0)


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    """Toy corpus + tokenizer + tokenized shards, shared by data/training tests."""
    import subprocess
    d = tmp_path_factory.mktemp("toy")
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([py, os.path.join(ROOT, "scripts", "make_toy_corpus.py"), "--out", str(d), "--n-train", "1500",
                    "--n-valid", "100"], check=True, env=env, capture_output=True)
    subprocess.run([py, "-m", "tinygpt.tokenizer.train", "--input", str(d / "train.txt"), "--vocab-size", "400",
                    "--out", str(d / "tok.json")], check=True, env=env, capture_output=True, cwd=ROOT)
    subprocess.run([py, "-m", "tinygpt.data.prepare", "--tok", str(d / "tok.json"), "--train", str(d / "train.txt"),
                    "--valid", str(d / "valid.txt"), "--out", str(d / "shards"), "--workers", "1"],
                   check=True, env=env, capture_output=True, cwd=ROOT)
    return d
