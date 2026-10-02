import json
import os

import numpy as np
import torch

from tinygpt.data.loader import TokenData
from tinygpt.data.prepare import iter_stories
from tinygpt.tokenizer.bpe import BPETokenizer


def test_shards_and_meta_consistent(toy_dir):
    d = toy_dir / "shards"
    meta = json.load(open(d / "meta.json"))
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    assert len(train) == meta["train_tokens"] and len(val) == meta["val_tokens"]
    assert int(train.max()) < meta["vocab_size"]
    assert int((train == meta["eos_id"]).sum()) == meta["train_stories"]   # exactly one EOS per story
    assert len(meta["train_sha256"]) == 64


def test_data_decodes_back_to_stories(toy_dir):
    tok = BPETokenizer.load(str(toy_dir / "tok.json"))
    val = np.memmap(toy_dir / "shards" / "val.bin", dtype=np.uint16, mode="r")
    first_story = val[: int(np.argmax(val == tok.eos_id))].astype(int).tolist()
    original = next(iter_stories(str(toy_dir / "valid.txt")))
    assert tok.decode(first_story) == original


def test_get_batch_shapes_and_shift(toy_dir):
    data = TokenData(str(toy_dir / "shards"))
    rng = np.random.default_rng(0)
    x, y = data.get_batch("train", 8, 32, rng)
    assert x.shape == y.shape == (8, 32) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])          # targets are inputs shifted by one


def test_val_batches_are_fixed(toy_dir):
    data = TokenData(str(toy_dir / "shards"))
    a = [x for x, _ in data.val_batches(3, 4, 16)]
    b = [x for x, _ in data.val_batches(3, 4, 16)]
    assert all(torch.equal(i, j) for i, j in zip(a, b))


def test_sequential_val_covers_set(toy_dir):
    data = TokenData(str(toy_dir / "shards"))
    n = sum(x.numel() for x, _ in data.sequential_val(16, 32))
    assert n == ((len(data.val) - 1) // 32) * 32


def test_multiprocess_prepare_matches_single(toy_dir, tmp_path):
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "s2"
    subprocess.run([sys.executable, "-m", "tinygpt.data.prepare", "--tok", str(toy_dir / "tok.json"),
                    "--train", str(toy_dir / "train.txt"), "--valid", str(toy_dir / "valid.txt"),
                    "--out", str(out), "--workers", "2"], check=True, cwd=root, capture_output=True)
    m1 = json.load(open(toy_dir / "shards" / "meta.json"))
    m2 = json.load(open(out / "meta.json"))
    assert m1["train_sha256"] == m2["train_sha256"]   # identical output regardless of worker count
