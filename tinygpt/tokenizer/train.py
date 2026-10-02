"""Train the BPE tokenizer.

    python -m tinygpt.tokenizer.train --input data/raw/TinyStoriesV2-GPT4-train.txt \
        --max-bytes 200000000 --vocab-size 4096 --out data/tok/tokenizer.json
"""
from __future__ import annotations

import argparse
import time
from typing import Iterator, List

from tinygpt.tokenizer.bpe import BPETokenizer


def iter_text_chunks(paths: List[str], max_bytes: int, chunk_chars: int = 8_000_000) -> Iterator[str]:
    """Yield text chunks from the start of each file until ~max_bytes characters were produced."""
    budget = max_bytes
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while budget > 0:
                chunk = f.read(min(chunk_chars, budget))
                if not chunk:
                    break
                budget -= len(chunk)
                yield chunk
        if budget <= 0:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="text file(s), stories separated by <|endoftext|>")
    ap.add_argument("--max-bytes", type=int, default=200_000_000, help="only read this many characters (RAM-safe)")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="data/tok/tokenizer.json")
    args = ap.parse_args()

    t0 = time.time()
    tok = BPETokenizer.train(iter_text_chunks(args.input, args.max_bytes), args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"[bpe] vocab_size={tok.vocab_size}  eos_id={tok.eos_id}  saved -> {args.out}  ({time.time() - t0:.0f}s)")

    sample = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(sample)
    print(f"[bpe] sample: {len(sample)} chars -> {len(ids)} tokens; round-trip ok = {tok.decode(ids) == sample}")


if __name__ == "__main__":
    main()
