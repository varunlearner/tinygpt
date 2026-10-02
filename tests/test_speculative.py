import copy

import pytest
import torch

from tinygpt.infer.generate import generate_cached
from tinygpt.infer.sampling import SamplingConfig, probs_from_logits
from tinygpt.infer.speculative import generate_speculative
from tests.conftest import make_model

GREEDY = SamplingConfig(temperature=0.0)


def _pair(vocab=64):
    target = make_model(seed=0, vocab=vocab, std_scale=3.0)
    draft = make_model(seed=1, vocab=vocab, d=32, layers=2, heads=2, kv=2, hidden=64, std_scale=3.0)
    return target, draft


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_greedy_identical_to_target_only(k):
    """T9: under greedy decoding, speculative output == target-only output, token for token."""
    target, draft = _pair()
    accepted_any_but_not_all = False
    for seed in range(12):
        torch.manual_seed(seed)
        prompt = torch.randint(0, 64, (1, 3 + seed % 6))
        ref = generate_cached(target, prompt, 45, GREEDY)
        out, st = generate_speculative(target, draft, prompt, 45, k, GREEDY)
        assert torch.equal(out, ref), f"k={k} seed={seed}"
        accepted_any_but_not_all |= 0 < st.acceptance_rate < 1
    assert accepted_any_but_not_all, "test never exercised the rejection path"


def test_all_accepted_path_identical_draft():
    """Draft == target: everything is accepted (the draft cache is one entry short each round)."""
    target, _ = _pair()
    draft = copy.deepcopy(target)
    prompt = torch.randint(0, 64, (1, 5))
    ref = generate_cached(target, prompt, 50, GREEDY)
    out, st = generate_speculative(target, draft, prompt, 50, 4, GREEDY)
    assert torch.equal(out, ref)
    assert st.acceptance_rate == 1.0
    assert st.mean_tokens_per_round > 4


def test_edge_cases():
    """T11: k=1, generation shorter than k, single new token, context boundary, single-token prompt."""
    target, draft = _pair()
    for prompt_len, new, k in [(1, 10, 3), (4, 1, 4), (4, 2, 4), (4, 3, 8), (10, 118, 5), (127, 1, 3)]:
        prompt = torch.randint(0, 64, (1, prompt_len))
        ref = generate_cached(target, prompt, new, GREEDY)
        out, _ = generate_speculative(target, draft, prompt, new, k, GREEDY)
        assert torch.equal(out, ref), (prompt_len, new, k)
        assert out.shape[1] == prompt_len + new


def test_eos_inside_draft_block():
    target, draft = _pair()
    prompt = torch.randint(0, 64, (1, 5))
    ref = generate_cached(target, prompt, 40, GREEDY)
    eos = int(ref[0, 5 + 6])                          # some token that appears as the 7th generated token
    ref_eos = generate_cached(target, prompt, 40, GREEDY, eos_id=eos)
    out, _ = generate_speculative(target, draft, prompt, 40, 5, GREEDY, eos_id=eos)
    assert torch.equal(out, ref_eos)
    assert out[0, -1].item() == eos


def test_context_overflow_rejected():
    target, draft = _pair()
    with pytest.raises(ValueError):
        generate_speculative(target, draft, torch.zeros(1, 100, dtype=torch.long), 40, 4, GREEDY)


def test_stats_are_consistent():
    target, draft = _pair()
    prompt = torch.randint(0, 64, (1, 5))
    out, st = generate_speculative(target, draft, prompt, 60, 4, GREEDY)
    assert st.generated == 60
    assert 0 <= st.accepted <= st.proposed
    assert st.rounds >= 60 / 5                         # at most k+1 tokens per round


# --------------------------------------------------------------------------------------------
# T10: the sampled output distribution must equal the target's (statistical test)
# --------------------------------------------------------------------------------------------

def _chi2_p(counts, expected):
    """Pearson chi-square p-value.

    Outcomes with zero probability must never occur; bins with expected count < 5 are pooled.
    """
    counts, expected = torch.as_tensor(counts, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    impossible = expected <= 0
    assert counts[impossible].sum() == 0, "sampled an outcome the target assigns zero probability"
    counts, expected = counts[~impossible], expected[~impossible]
    small = expected < 5
    if small.any():
        counts = torch.cat([counts[~small], counts[small].sum().reshape(1)])
        expected = torch.cat([expected[~small], expected[small].sum().reshape(1)])
    stat = ((counts - expected) ** 2 / expected).sum()
    df = len(expected) - 1
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), stat / 2))


def _exact_joint(target, prompt, cfg, V):
    """Exact P(x1, x2 | prompt) under the target model with the given sampling config."""
    with torch.no_grad():
        p1 = probs_from_logits(target(prompt)[0, -1], cfg)
        joint = torch.zeros(V, V, dtype=torch.float64)
        for a in range(V):
            ext = torch.cat([prompt, torch.tensor([[a]])], dim=1)
            joint[a] = p1[a].double() * probs_from_logits(target(ext)[0, -1], cfg).double()
    return p1.double(), joint


def _empirical(sampler, n, V):
    first = torch.zeros(V)
    joint = torch.zeros(V, V)
    for i in range(n):
        toks = sampler(i)[0, -3:]                       # last 3 generated tokens; use the first two
        first[toks[0]] += 1
        joint[toks[0], toks[1]] += 1
    return first, joint


@pytest.mark.parametrize("temperature,top_k", [(1.0, 0), (0.8, 3)])
def test_speculative_preserves_target_distribution(temperature, top_k):
    V = 5
    target = make_model(seed=5, vocab=V, d=16, layers=2, heads=2, kv=2, hidden=32, max_len=32, std_scale=5.0)
    draft = make_model(seed=6, vocab=V, d=16, layers=1, heads=2, kv=2, hidden=32, max_len=32, std_scale=5.0)
    cfg = SamplingConfig(temperature=temperature, top_k=top_k)
    prompt = torch.tensor([[1, 3, 2]])
    p1, joint = _exact_joint(target, prompt, cfg, V)

    with torch.no_grad():                                # the draft must genuinely differ from the target
        q1 = probs_from_logits(draft(prompt)[0, -1], cfg).double()
    assert 0.5 * (p1 - q1).abs().sum() > 0.05, "toy models too similar: test would have no power"

    n = 3000
    gen = torch.Generator().manual_seed(123)
    first, jnt = _empirical(lambda i: generate_speculative(target, draft, prompt, 3, 3, cfg, generator=gen)[0], n, V)
    assert _chi2_p(first, p1 * n) > 1e-3
    assert _chi2_p(jnt.flatten(), joint.flatten() * n) > 1e-3


def test_distribution_test_has_power():
    """Sampling from the DRAFT instead of the target must be detected as wrong by the same test."""
    V = 5
    target = make_model(seed=5, vocab=V, d=16, layers=2, heads=2, kv=2, hidden=32, max_len=32, std_scale=5.0)
    draft = make_model(seed=6, vocab=V, d=16, layers=1, heads=2, kv=2, hidden=32, max_len=32, std_scale=5.0)
    cfg = SamplingConfig(temperature=1.0)
    prompt = torch.tensor([[1, 3, 2]])
    p1, joint = _exact_joint(target, prompt, cfg, V)
    n = 3000
    gen = torch.Generator().manual_seed(5)
    first, jnt = _empirical(lambda i: generate_cached(draft, prompt, 3, cfg, generator=gen), n, V)
    assert _chi2_p(jnt.flatten(), joint.flatten() * n) < 1e-6
