"""Tokenize text into uint16 memmap shards (train.bin / val.bin) + meta.json.

    python -m tinygpt.data.prepare --tok data/tok/tokenizer.json \
        --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt \
        --out data/shards --workers 4

Streams the input, so peak RAM stays small (a few hundred MB per worker).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import deque
from typing import Iterator, List, Tuple

import numpy as np

from tinygpt.tokenizer.bpe import EOS_TOKEN, BPETokenizer

_TOK: BPETokenizer | None = None


def iter_stories(path: str, chunk_chars: int = 64_000_000, min_chars: int = 50) -> Iterator[str]:
    """Yield stories from a text file whose stories are separated by <|endoftext|>."""
    carry = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            parts = (carry + chunk).split(EOS_TOKEN)
            carry = parts.pop()  # last part may be an incomplete story
            for p in parts:
                s = p.strip()
                if len(s) >= min_chars:
                    yield s
    s = carry.strip()
    if len(s) >= min_chars:
        yield s


def batch_stories(stories: Iterator[str], batch_chars: int = 2_000_000) -> Iterator[List[str]]:
    batch, size = [], 0
    for s in stories:
        batch.append(s)
        size += len(s)
        if size >= batch_chars:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def _init_worker(tok_path: str) -> None:
    global _TOK
    _TOK = BPETokenizer.load(tok_path)


def _encode_batch(stories: List[str]) -> Tuple[np.ndarray, int, int]:
    assert _TOK is not None
    eos = _TOK.eos_id
    out: List[int] = []
    n_bytes = 0
    for s in stories:
        out.extend(_TOK.encode(s, allow_special=False))
        out.append(eos)
        n_bytes += len(s.encode("utf-8"))
    return np.asarray(out, dtype=np.uint16), n_bytes, len(stories)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def encode_file(txt_path: str, out_path: str, tok_path: str, workers: int) -> dict:
    t0 = time.time()
    n_tokens = n_bytes = n_stories = 0
    batches = batch_stories(iter_stories(txt_path))

    def write(fh, res):
        nonlocal n_tokens, n_bytes, n_stories
        arr, nb, ns = res
        fh.write(arr.tobytes())
        n_tokens += len(arr)
        n_bytes += nb
        n_stories += ns
        if n_stories % 20000 < ns:
            rate = n_tokens / max(time.time() - t0, 1e-9)
            print(f"\r[prepare] {os.path.basename(out_path)}: {n_stories:,} stories  {n_tokens / 1e6:,.1f}M tokens  "
                  f"({rate / 1e3:,.0f}k tok/s)", end="", flush=True)

    with open(out_path, "wb") as fh:
        if workers <= 1:
            _init_worker(tok_path)
            for b in batches:
                write(fh, _encode_batch(b))
        else:
            ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
                pending: deque = deque()
                for b in batches:
                    pending.append(pool.apply_async(_encode_batch, (b,)))
                    if len(pending) >= workers * 2:  # bounded queue -> bounded RAM
                        write(fh, pending.popleft().get())
                while pending:
                    write(fh, pending.popleft().get())
    print(f"\n[prepare] {out_path}: {n_stories:,} stories, {n_tokens:,} tokens, "
          f"{n_bytes / max(n_tokens, 1):.2f} bytes/token ({time.time() - t0:.0f}s)")
    return {"tokens": n_tokens, "bytes": n_bytes, "stories": n_stories}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="data/tok/tokenizer.json")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = BPETokenizer.load(args.tok)
    assert tok.vocab_size < 65536, "uint16 storage needs vocab < 65536"
    tr = encode_file(args.train, os.path.join(args.out, "train.bin"), args.tok, args.workers)
    va = encode_file(args.valid, os.path.join(args.out, "val.bin"), args.tok, args.workers)
    meta = {
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "train_tokens": tr["tokens"], "train_bytes": tr["bytes"], "train_stories": tr["stories"],
        "val_tokens": va["tokens"], "val_bytes": va["bytes"], "val_stories": va["stories"],
        "train_sha256": sha256_file(os.path.join(args.out, "train.bin")),
        "val_sha256": sha256_file(os.path.join(args.out, "val.bin")),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
