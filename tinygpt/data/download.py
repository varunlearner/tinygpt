"""Download TinyStories (V2, GPT-4 stories) from Hugging Face.

    python -m tinygpt.data.download --out data/raw                 # full train (~2.2 GB) + valid
    python -m tinygpt.data.download --out data/raw --train-mb 300  # only the first 300 MB of train

Supports resuming an interrupted download. Dataset license: CDLA-Sharing-1.0 (see the dataset card).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}


def download(url: str, dest: str, max_bytes: int | None = None, chunk: int = 1 << 20) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {"User-Agent": "tinygpt/0.1"}
    if max_bytes is not None and have >= max_bytes:
        print(f"[download] {dest} already has >= {max_bytes / 1e6:.0f} MB, skipping")
        return
    if have:
        headers["Range"] = f"bytes={have}-" if max_bytes is None else f"bytes={have}-{max_bytes - 1}"
    elif max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            print(f"[download] {dest} already complete")
            return
        raise
    mode = "ab" if have and resp.status == 206 else "wb"
    if mode == "wb":
        have = 0
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    t0, got = time.time(), have
    with open(dest, mode) as f:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if int((time.time() - t0) * 2) % 2 == 0:
                pct = f"{100 * got / total:5.1f}%" if total else ""
                print(f"\r[download] {os.path.basename(dest)}: {got / 1e6:8.1f} MB {pct}", end="", file=sys.stderr)
    print(f"\n[download] saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--train-mb", type=int, default=0, help="download only the first N MB of train (0 = all)")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    download(BASE + FILES["valid"], os.path.join(args.out, FILES["valid"]))
    if not args.skip_train:
        limit = args.train_mb * 1_000_000 if args.train_mb > 0 else None
        download(BASE + FILES["train"], os.path.join(args.out, FILES["train"]), max_bytes=limit)


if __name__ == "__main__":
    main()
