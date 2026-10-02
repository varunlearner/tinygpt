import pytest
import torch

from tinygpt.infer.generate import generate_cached, generate_naive
from tinygpt.infer.sampling import SamplingConfig
from tinygpt.model.kv_cache import KVCache
from tests.conftest import make_model

GREEDY = SamplingConfig(temperature=0.0)


@pytest.mark.parametrize("fixture", ["small_model", "gqa_model"])
def test_prefill_plus_decode_equals_full_forward(request, fixture):
    """T6: KV-cache logits match the no-cache forward (fp32)."""
    m = request.getfixturevalue(fixture)
    x = torch.randint(0, 64, (2, 30))
    full = m(x)
    cache = KVCache.for_model(m, 2)
    outs = [m(x[:, :11], cache=cache)]
    for t in range(11, 30):
        outs.append(m(x[:, t:t + 1], cache=cache))
    assert (full - torch.cat(outs, dim=1)).abs().max().item() < 1e-4
    assert cache.length == 30


def test_chunked_append_equals_full_forward(small_model):
    x = torch.randint(0, 64, (1, 25))
    full = small_model(x)
    cache = KVCache.for_model(small_model, 1)
    parts = [small_model(x[:, a:b], cache=cache) for a, b in [(0, 5), (5, 6), (6, 15), (15, 25)]]
    assert (full - torch.cat(parts, dim=1)).abs().max().item() < 1e-4


def test_truncate_then_redecode_reproduces_logits(small_model):
    """T7: rolling back and re-decoding gives the same logits as the first time."""
    x = torch.randint(0, 64, (1, 20))
    cache = KVCache.for_model(small_model, 1)
    small_model(x[:, :10], cache=cache)
    first = small_model(x[:, 10:15], cache=cache)
    cache.truncate(10)
    assert cache.length == 10
    # overwrite with DIFFERENT tokens first, then roll back again: stale data must not leak
    small_model(torch.randint(0, 64, (1, 5)), cache=cache)
    cache.truncate(10)
    again = small_model(x[:, 10:15], cache=cache)
    assert torch.allclose(first, again, atol=1e-5)


def test_batched_rows_match_single_sequences(small_model):
    """T8: each row of a batched cached decode equals decoding that sequence alone."""
    prompts = torch.randint(0, 64, (3, 8))
    batched = generate_cached(small_model, prompts, 20, GREEDY)
    for i in range(3):
        single = generate_cached(small_model, prompts[i:i + 1], 20, GREEDY)
        assert torch.equal(batched[i:i + 1], single)


@pytest.mark.parametrize("fixture", ["small_model", "gqa_model"])
def test_greedy_cached_equals_naive(request, fixture):
    m = request.getfixturevalue(fixture)
    for seed in range(5):
        torch.manual_seed(seed)
        p = torch.randint(0, 64, (1, 6 + seed))
        assert torch.equal(generate_naive(m, p, 40, GREEDY), generate_cached(m, p, 40, GREEDY))


def test_sampling_reproducible_with_seed(small_model):
    p = torch.randint(0, 64, (2, 5))
    cfg = SamplingConfig(temperature=0.9, top_k=20, top_p=0.95)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(generate_cached(small_model, p, 25, cfg, generator=g1),
                       generate_cached(small_model, p, 25, cfg, generator=g2))


def test_cache_overflow_raises(small_model):
    cache = KVCache.for_model(small_model, 1, max_len=8)
    with pytest.raises(RuntimeError):
        small_model(torch.zeros(1, 9, dtype=torch.long), cache=cache)


def test_eos_stops_generation(small_model):
    p = torch.randint(0, 64, (1, 5))
    out = generate_cached(small_model, p, 30, GREEDY)
    eos = int(out[0, 8])                               # pretend the 4th generated token is EOS
    stopped = generate_cached(small_model, p, 30, GREEDY, eos_id=eos)
    first = int((out[0, 5:] == eos).nonzero()[0])
    assert stopped.shape[1] == 5 + first + 1
    assert stopped[0, -1].item() == eos


def test_cache_memory_formula(small_model):
    cfg = small_model.cfg
    cache = KVCache.for_model(small_model, 3, max_len=100)
    per_token = cfg.kv_bytes_per_token(4)              # fp32 model
    assert cache.nbytes == per_token * 3 * 100
