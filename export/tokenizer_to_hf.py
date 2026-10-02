"""Convert our BPE tokenizer to a Hugging Face `tokenizer.json` (byte-level BPE, GPT-2 style).

    python -m export.tokenizer_to_hf --tok data/tok/tokenizer.json --out export/hf

The HF tokenizer reproduces our ids exactly (verified in tests/test_export.py).
"""
from __future__ import annotations

import argparse
import json
import os

from tinygpt.tokenizer.bpe import BPETokenizer


def bytes_to_unicode() -> dict:
    """GPT-2's reversible byte -> printable-unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def build_hf_tokenizer(tok: BPETokenizer):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    b2u = bytes_to_unicode()

    def tstr(i: int) -> str:
        return "".join(b2u[b] for b in tok.vocab[i])

    vocab = {tstr(i): i for i in range(256)}
    for name, i in tok.special_tokens.items():
        vocab[name] = i
    for i in range(tok.first_merge_id, tok.vocab_size):
        vocab[tstr(i)] = i
    merges = [(tstr(a), tstr(b)) for a, b in tok.merges]

    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([AddedToken(name, special=True) for name in tok.special_tokens])
    return tk


def export_tokenizer(tok_path: str, out_dir: str) -> None:
    tok = BPETokenizer.load(tok_path)
    os.makedirs(out_dir, exist_ok=True)
    build_hf_tokenizer(tok).save(os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|endoftext|>",
                   "pad_token": "<|pad|>", "model_max_length": 512, "clean_up_tokenization_spaces": False}, f, indent=2)
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as f:
        json.dump({"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}, f, indent=2)
    print(f"[export] tokenizer -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="data/tok/tokenizer.json")
    ap.add_argument("--out", default="export/hf")
    args = ap.parse_args()
    export_tokenizer(args.tok, args.out)


if __name__ == "__main__":
    main()
