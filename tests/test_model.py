import pytest
import torch

from tinygpt.model.config import GPTConfig
from tinygpt.model.gpt import GPT, apply_rope, build_rope
from tests.conftest import make_model


@pytest.mark.parametrize("kw", [dict(), dict(n_kv_heads=2), dict(d_model=640, n_heads=10, n_kv_heads=10, mlp_hidden=1728),
                                dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=4, mlp_hidden=704)])
def test_param_count_formula_matches(kw):
    cfg = GPTConfig(**kw)
    assert GPT(cfg).num_params() == cfg.num_params()


def test_target_and_draft_sizes_match_design_doc():
    assert abs(GPTConfig().num_params() / 1e6 - 27.8) < 0.2
    assert abs(GPTConfig(d_model=256, n_layers=4, n_heads=4, n_kv_heads=4, mlp_hidden=704).num_params() / 1e6 - 4.26) < 0.1
    assert GPTConfig().kv_bytes_per_token(2) == 16384   # 16 KB / token in fp16


def test_forward_shapes(small_model):
    x = torch.randint(0, 64, (3, 17))
    assert small_model(x).shape == (3, 17, 64)
    assert small_model(x, logits_last=1).shape == (3, 1, 64)


def test_tied_embeddings():
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight


def test_causality(small_model):
    """Changing a token at position t must not change logits at positions < t."""
    x = torch.randint(0, 64, (2, 24))
    y = x.clone()
    y[:, 15] = (y[:, 15] + 7) % 64
    a, b = small_model(x), small_model(y)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-6)
    assert not torch.allclose(a[:, 15:], b[:, 15:], atol=1e-6)


def test_rope_offset_matches_full_pass(small_model):
    """Logits at position t are the same whether computed in a full pass or as one token at start_pos=t."""
    from tinygpt.model.kv_cache import KVCache
    x = torch.randint(0, 64, (1, 20))
    full = small_model(x)
    cache = KVCache.for_model(small_model, 1)
    small_model(x[:, :19], cache=cache)
    last = small_model(x[:, 19:20], cache=cache)
    assert torch.allclose(full[:, 19], last[:, 0], atol=1e-4)


def test_rope_is_rotation():
    cos, sin = build_rope(16, 32, 10000.0)
    x = torch.randn(1, 2, 5, 16)
    y = apply_rope(x, cos[:5], sin[:5])
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)   # rotations preserve norms


def test_gqa_shapes(gqa_model):
    x = torch.randint(0, 64, (2, 10))
    assert gqa_model(x).shape == (2, 10, 64)
    assert gqa_model.blocks[0].attn.wk.weight.shape[0] == 2 * gqa_model.cfg.head_dim


def test_too_long_sequence_raises(small_model):
    with pytest.raises(ValueError):
        small_model(torch.zeros(1, 129, dtype=torch.long))


def test_half_precision_keeps_fp32_rope():
    m = make_model().half()
    assert m.rope_cos.dtype == torch.float32


def test_overfit_single_batch():
    """Sanity: the model + optimizer can memorize one batch (catches broken gradients / masks)."""
    torch.manual_seed(0)
    m = make_model(seed=0)
    m.train()
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x).view(-1, 64), y.reshape(-1))
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 3.5 and loss.item() < 0.3
