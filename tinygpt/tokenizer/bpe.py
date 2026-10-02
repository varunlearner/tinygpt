"""Byte-level BPE tokenizer, written from scratch (no tokenizer libraries).

Vocabulary layout (ids):
    0..255      the 256 raw byte values
    256, 257    special tokens  <|endoftext|>, <|pad|>
    258..       learned merges, in merge order

So vocab_size = 258 + n_merges  (4096 -> 3838 merges).

Training is fast because merges are computed over the table of *unique pre-tokenized words*
(with counts) instead of the raw text, and pair statistics are updated incrementally.
"""
from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import regex

# GPT-2 style pre-tokenization: merges never cross these boundaries.
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
DEFAULT_SPECIALS = [EOS_TOKEN, PAD_TOKEN]
N_BYTES = 256
FORMAT = "tinygpt-bpe-v1"


def _special_regex(specials: Sequence[str]) -> regex.Pattern:
    return regex.compile("|".join(regex.escape(s) for s in specials))


def merge_ids(ids: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every non-overlapping (left-to-right) occurrence of `pair` in ids by new_id."""
    a, b = pair
    out: List[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_words(texts: Iterable[str], pattern: str = GPT2_PATTERN,
                specials: Sequence[str] = DEFAULT_SPECIALS) -> Dict[bytes, int]:
    """Pre-tokenize texts and count each distinct word (as UTF-8 bytes)."""
    pat = regex.compile(pattern)
    sp = _special_regex(specials)
    counts: Counter = Counter()
    for text in texts:
        for piece in sp.split(text):
            counts.update(pat.findall(piece))
    return {w.encode("utf-8"): c for w, c in counts.items()}


def train_bpe(word_counts: Dict[bytes, int], n_merges: int, first_id: int,
              verbose: bool = False) -> List[Tuple[int, int]]:
    """Learn `n_merges` merges from a {word_bytes: count} table.

    Ties are broken by the smallest (a, b) pair, so training is fully deterministic.
    """
    words: List[List[int]] = [list(w) for w in word_counts.keys()]
    freqs: List[int] = list(word_counts.values())

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_words: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_words[p].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges: List[Tuple[int, int]] = []

    for m in range(n_merges):
        best = None
        while heap:  # lazy deletion: skip stale heap entries
            neg_c, p = heapq.heappop(heap)
            c = pair_counts.get(p, 0)
            if c > 0 and c == -neg_c:
                best = p
                break
        if best is None:
            if verbose:
                print(f"[bpe] ran out of pairs after {m} merges")
            break
        new_id = first_id + m
        merges.append(best)

        changed = set()
        for wi in list(pair_words[best]):
            w = words[wi]
            if not any(x == best[0] and y == best[1] for x, y in zip(w, w[1:])):
                continue  # stale membership
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                changed.add(p)
            w = merge_ids(w, best, new_id)
            words[wi] = w
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_words[p].add(wi)
                changed.add(p)
        for p in changed:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, (-c, p))
        pair_words.pop(best, None)

        if verbose and (m + 1) % 500 == 0:
            print(f"[bpe] {m + 1}/{n_merges} merges")
    return merges


class BPETokenizer:
    def __init__(self, merges: Sequence[Tuple[int, int]],
                 special_tokens: Optional[Sequence[str]] = None,
                 pattern: str = GPT2_PATTERN):
        specials = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIALS)
        self.pattern = pattern
        self.special_tokens: Dict[str, int] = {s: N_BYTES + i for i, s in enumerate(specials)}
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        self.first_merge_id = N_BYTES + len(specials)
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]
        self.merge_ranks = {p: i for i, p in enumerate(self.merges)}

        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[self.first_merge_id + i] = self.vocab[a] + self.vocab[b]

        self._pat = regex.compile(pattern)
        self._special_split = regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
        self._cache: Dict[bytes, List[int]] = {}

    # ---- properties -------------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.first_merge_id + len(self.merges)

    @property
    def eos_id(self) -> int:
        return self.special_tokens[EOS_TOKEN]

    @property
    def pad_id(self) -> int:
        return self.special_tokens[PAD_TOKEN]

    # ---- training ---------------------------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              special_tokens: Optional[Sequence[str]] = None,
              pattern: str = GPT2_PATTERN, verbose: bool = False) -> "BPETokenizer":
        specials = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIALS)
        first_id = N_BYTES + len(specials)
        n_merges = vocab_size - first_id
        if n_merges <= 0:
            raise ValueError(f"vocab_size must be > {first_id}")
        if verbose:
            print("[bpe] counting words ...")
        wc = count_words(texts, pattern, specials)
        if verbose:
            print(f"[bpe] {len(wc):,} unique words, {sum(wc.values()):,} total; learning {n_merges} merges")
        merges = train_bpe(wc, n_merges, first_id, verbose)
        return cls(merges, specials, pattern)

    # ---- encode / decode --------------------------------------------------------------------
    def _encode_word(self, word: bytes) -> List[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word)
        while len(ids) > 1:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                r = self.merge_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = merge_ids(ids, best_pair, self.first_merge_id + best_rank)
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        out: List[int] = []
        pieces = self._special_split.split(text) if allow_special else [text]
        for piece in pieces:
            if allow_special and piece in self.special_tokens:
                out.append(self.special_tokens[piece])
                continue
            for w in self._pat.findall(piece):
                out.extend(self._encode_word(w.encode("utf-8")))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: List[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                # ids outside the vocabulary (e.g. unused embedding rows of an over-sized model) -> U+FFFD
                parts.append(self.vocab.get(i, b"\xef\xbf\xbd"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        if i in self.id_to_special:
            return self.id_to_special[i].encode("utf-8")
        return self.vocab[i]

    # ---- persistence ------------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        obj = {
            "format": FORMAT,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        specials = [k for k, _ in sorted(obj["special_tokens"].items(), key=lambda kv: kv[1])]
        return cls([tuple(m) for m in obj["merges"]], specials, obj["pattern"])
