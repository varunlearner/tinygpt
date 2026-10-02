"""T2: compare our BPE tokenizer with a Hugging Face `tokenizers` ByteLevelBPE trained on the same data.

    pip install tokenizers
    python scripts/tokenizer_parity.py --train data/raw/TinyStoriesV2-GPT4-train.txt \
        --valid data/raw/TinyStoriesV2-GPT4-valid.txt --tok data/tok/tokenizer.json --max-bytes 200000000

Reports bytes/token on the validation set for both (higher = better compression). Target: within 2%.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.tokenizer.bpe import BPETokenizer  # noqa: E402
from tinygpt.tokenizer.train import iter_text_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--tok", default="data/tok/tokenizer.json")
    ap.add_argument("--max-bytes", type=int, default=200_000_000)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    args = ap.parse_args()
    from tokenizers import ByteLevelBPETokenizer

    ours = BPETokenizer.load(args.tok)
    hf = ByteLevelBPETokenizer()
    hf.train_from_iterator(iter_text_chunks([args.train], args.max_bytes), vocab_size=args.vocab_size,
                           special_tokens=["<|endoftext|>", "<|pad|>"], show_progress=False)
    text = next(iter_text_chunks([args.valid], args.val_chars, chunk_chars=args.val_chars))
    n_bytes = len(text.encode("utf-8"))
    n_ours = len(ours.encode(text, allow_special=True))
    n_hf = len(hf.encode(text).ids)
    bpt_ours, bpt_hf = n_bytes / n_ours, n_bytes / n_hf
    print(f"validation text: {n_bytes:,} bytes")
    print(f"ours : vocab {ours.vocab_size}  {n_ours:,} tokens  {bpt_ours:.3f} bytes/token")
    print(f"HF   : vocab {hf.get_vocab_size()}  {n_hf:,} tokens  {bpt_hf:.3f} bytes/token")
    rel = 100 * (bpt_ours / bpt_hf - 1)
    print(f"difference: {rel:+.2f}%  ->  {'PASS' if abs(rel) <= 2 else 'CHECK (>2%)'}")


if __name__ == "__main__":
    main()
