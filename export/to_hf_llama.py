"""Export a trained model to Hugging Face `LlamaForCausalLM` (+ tokenizer) with a logit-parity check.

    python -m export.to_hf_llama --ckpt runs/target_28m/model.pt --tok data/tok/tokenizer.json --out export/hf

The architecture (RoPE rotate-half, RMSNorm, SwiGLU, GQA) already matches Llama, so weights are only renamed.
The result loads with `AutoModelForCausalLM.from_pretrained(...)`, converts to GGUF for llama.cpp, and to
OpenVINO via optimum-intel.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from tinygpt.model.gpt import GPT
from tinygpt.model.io import load_model


def to_hf_model(model: GPT, eos_id: Optional[int] = None, pad_id: Optional[int] = None):
    from transformers import LlamaConfig, LlamaForCausalLM
    c = model.cfg
    hf_cfg = LlamaConfig(
        vocab_size=c.vocab_size, hidden_size=c.d_model, intermediate_size=c.mlp_hidden,
        num_hidden_layers=c.n_layers, num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads,
        head_dim=c.head_dim, max_position_embeddings=c.max_seq_len, rms_norm_eps=c.norm_eps,
        rope_theta=c.rope_theta, tie_word_embeddings=c.tie_embeddings, attention_bias=False, mlp_bias=False,
        hidden_act="silu", bos_token_id=None, eos_token_id=eos_id, pad_token_id=pad_id,
    )
    hf = LlamaForCausalLM(hf_cfg)
    sd = {"model.embed_tokens.weight": model.tok_emb.weight.detach().float(),
          "model.norm.weight": model.norm.weight.detach().float()}
    for i, b in enumerate(model.blocks):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = b.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = b.mlp_norm.weight
        sd[p + "self_attn.q_proj.weight"] = b.attn.wq.weight
        sd[p + "self_attn.k_proj.weight"] = b.attn.wk.weight
        sd[p + "self_attn.v_proj.weight"] = b.attn.wv.weight
        sd[p + "self_attn.o_proj.weight"] = b.attn.wo.weight
        sd[p + "mlp.gate_proj.weight"] = b.mlp.w_gate.weight
        sd[p + "mlp.up_proj.weight"] = b.mlp.w_up.weight
        sd[p + "mlp.down_proj.weight"] = b.mlp.w_down.weight
    if not c.tie_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight
    sd = {k: v.detach().float().clone() for k, v in sd.items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and c.tie_embeddings)]
    if missing or unexpected:
        raise RuntimeError(f"state-dict mismatch: missing={missing} unexpected={unexpected}")
    if c.tie_embeddings:
        hf.tie_weights()
    return hf.eval()


@torch.no_grad()
def max_logit_diff(native: GPT, hf, n_seqs: int = 4, length: int = 48, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, native.cfg.vocab_size, (n_seqs, min(length, native.cfg.max_seq_len)), generator=g)
    a = native.float().eval()(ids)
    b = hf.float().eval()(input_ids=ids).logits
    return (a - b).abs().max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", default="data/tok/tokenizer.json")
    ap.add_argument("--out", default="export/hf")
    args = ap.parse_args()

    model = load_model(args.ckpt, "cpu", torch.float32)
    eos = pad = None
    if os.path.exists(args.tok):
        from tinygpt.tokenizer.bpe import BPETokenizer
        t = BPETokenizer.load(args.tok)
        eos, pad = t.eos_id, t.pad_id
    hf = to_hf_model(model, eos, pad)
    diff = max_logit_diff(model, hf)
    print(f"[export] max |logit diff| native vs HF Llama = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("parity check FAILED (> 1e-3); not writing the export")
    os.makedirs(args.out, exist_ok=True)
    hf.save_pretrained(args.out)
    if os.path.exists(args.tok):
        from export.tokenizer_to_hf import export_tokenizer
        export_tokenizer(args.tok, args.out)
    print(f"[export] HF Llama model -> {args.out}")


if __name__ == "__main__":
    main()
