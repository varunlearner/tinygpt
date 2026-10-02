import random

from tinygpt.tokenizer.bpe import DEFAULT_SPECIALS, BPETokenizer, count_words, merge_ids, train_bpe

CORPUS = ("The quick brown fox jumps over the lazy dog. Once upon a time there was a little girl. " * 60
          + "Numbers 12345, punctuation!? and\nnewlines\n\n plus  double  spaces. ") * 3


def _tok(vocab=340):
    return BPETokenizer.train([CORPUS], vocab)


def test_vocab_layout():
    t = _tok(340)
    assert t.vocab_size == 340
    assert t.eos_id == 256 and t.pad_id == 257
    assert t.first_merge_id == 258


def test_roundtrip_random_unicode_and_bytes():
    t = _tok()
    rng = random.Random(0)
    samples = ["", " ", "\n", "hello", "héllo wörld", "你好，世界", "emoji 🙂🚀 test", "tabs\tand\r\nwindows",
               "  leading and trailing  ", "don't stop, we'll go!"]
    for _ in range(200):
        n = rng.randint(1, 40)
        samples.append("".join(chr(rng.choice([rng.randint(32, 126), rng.randint(160, 900), rng.randint(0x4E00, 0x4E80)]))
                               for _ in range(n)))
    for s in samples:
        assert t.decode(t.encode(s)) == s, repr(s)


def test_special_tokens():
    t = _tok()
    ids = t.encode("Once upon a time<|endoftext|>The end")
    assert t.eos_id in ids and ids.count(t.eos_id) == 1
    assert t.decode(ids) == "Once upon a time<|endoftext|>The end"
    # with allow_special=False the marker is ordinary text
    plain = t.encode("<|endoftext|>", allow_special=False)
    assert t.eos_id not in plain and t.decode(plain) == "<|endoftext|>"


def test_save_load_identical(tmp_path):
    t = _tok()
    p = tmp_path / "tok.json"
    t.save(str(p))
    t2 = BPETokenizer.load(str(p))
    assert t2.merges == t.merges and t2.vocab_size == t.vocab_size
    assert t2.encode(CORPUS[:2000]) == t.encode(CORPUS[:2000])


def test_training_is_deterministic():
    a, b = _tok().merges, _tok().merges
    assert a == b


def test_encoding_matches_sequential_merge_application():
    """encode() (lowest-rank-pair-first) must equal applying the learned merges sequentially."""
    wc = count_words([CORPUS])
    first = 258
    merges = train_bpe(wc, 60, first)
    t = BPETokenizer(merges)
    for word in list(wc)[:80]:
        ids = list(word)
        for r, pair in enumerate(merges):
            ids = merge_ids(ids, pair, first + r)
        assert t._encode_word(word) == ids


def test_compresses_repetitive_text():
    t = _tok()
    ids = t.encode(CORPUS)
    assert len(ids) < 0.5 * len(CORPUS.encode())


def test_merge_ids_overlap():
    assert merge_ids([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_ids([1, 1, 1, 1], (1, 1), 9) == [9, 9]


def test_specials_default():
    assert len(DEFAULT_SPECIALS) == 2


def test_decode_tolerates_out_of_vocab_ids():
    t = _tok(340)
    assert "\ufffd" in t.decode([t.vocab_size + 50])
    assert t.decode(t.encode("hello") + [t.vocab_size + 1]).startswith("hello")
