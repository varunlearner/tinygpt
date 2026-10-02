import pytest
import torch

from tinygpt.tokenizer.bpe import BPETokenizer
from tests.conftest import make_model

TEXTS = ["Once upon a time, there was a little girl named Lily.", "  leading spaces and\nnewlines\n\n", "héllo wörld 你好 🙂",
         "Numbers 12345 and punctuation!?", "don't we'll they're", "", "a", "<|endoftext|>tail"]


def test_hf_tokenizer_matches_ours(toy_dir):
    pytest.importorskip("tokenizers")
    from export.tokenizer_to_hf import build_hf_tokenizer
    tok = BPETokenizer.load(str(toy_dir / "tok.json"))
    hf = build_hf_tokenizer(tok)
    corpus = open(toy_dir / "valid.txt", encoding="utf-8").read()[:3000]
    for t in TEXTS + [corpus]:
        assert hf.encode(t).ids == tok.encode(t), repr(t[:40])
        assert hf.decode(hf.encode(t).ids, skip_special_tokens=False) == t
    assert hf.token_to_id("<|endoftext|>") == tok.eos_id


@pytest.mark.parametrize("kv", [4, 2])
def test_hf_llama_logit_parity(kv):
    pytest.importorskip("transformers")
    from export.to_hf_llama import max_logit_diff, to_hf_model
    m = make_model(seed=2, kv=kv, std_scale=3.0, max_len=64)
    hf = to_hf_model(m, eos_id=1, pad_id=2)
    assert max_logit_diff(m, hf, length=40) < 1e-4


def test_hf_generate_matches_native_greedy():
    pytest.importorskip("transformers")
    from export.to_hf_llama import to_hf_model
    from tinygpt.infer.generate import generate_cached
    from tinygpt.infer.sampling import SamplingConfig
    m = make_model(seed=3, std_scale=3.0, max_len=64)
    hf = to_hf_model(m)
    p = torch.randint(0, 64, (1, 6))
    ref = generate_cached(m, p, 30, SamplingConfig(temperature=0.0))
    out = hf.generate(p, attention_mask=torch.ones_like(p), max_new_tokens=30, min_new_tokens=30, do_sample=False)
    assert torch.equal(ref, out)


def test_int8_quantization_close_to_fp32():
    from export.quantize import Int8Linear, linear_weight_bytes, quantize_model_int8
    m = make_model(seed=4, std_scale=2.0)
    q = quantize_model_int8(m)
    x = torch.randint(0, 64, (2, 24))
    a, b = m(x), q(x)
    rel = (a - b).norm() / a.norm()
    assert rel < 0.05, f"int8 logits too far from fp32: {rel:.3f}"
    assert any(isinstance(mod, Int8Linear) for mod in q.modules())
    assert linear_weight_bytes(q) < 0.30 * linear_weight_bytes(m)        # ~4x smaller (plus small scales)
    assert m.tok_emb.weight.dtype == q.tok_emb.weight.dtype              # embedding untouched
