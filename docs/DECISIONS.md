# Architecture Decision Records (ADRs)

**Last updated:** 2026-09-30

Each record: **Context** (the problem), **Decision**, **Consequences** (trade-offs, what would make us revisit). Add a new ADR whenever a choice would be expensive to reverse or hard to explain later. Never delete old ones — mark them *Superseded by ADR-XXX*.

Status values: Proposed · Accepted · Superseded.

---

## ADR-001 — Vocabulary size of 4096

**Status:** Accepted (revisit after tokenizer parity results)

**Context.** TinyStories uses a small, simple vocabulary of English words. Vocab size drives embedding parameters (`vocab × d_model`), softmax cost per step, and sequence length (fewer merges → more tokens per story).

**Decision.** 4096 tokens: 256 bytes + 2 specials + 3838 merges.

**Consequences.**
- Embeddings cost only ~2.1M parameters at d=512, leaving most of the ~28M budget for transformer blocks.
- A tiny LM head keeps the per-step softmax cheap, which matters for the speculative-decoding cost ratio `c`.
- Sequences are longer than with a 32k vocab; acceptable at ctx 512.
- Revisit with 8192 if bytes/token is poor or samples show frequent word-fragmenting.

---

## ADR-002 — Llama-style architecture (RoPE, RMSNorm, SwiGLU, no biases)

**Status:** Accepted

**Context.** GPT-2-style learned positional embeddings and LayerNorm+GELU are simpler, but a Llama-style model exports cleanly to Hugging Face `LlamaForCausalLM`, GGUF, and llama.cpp, and reflects current practice.

**Decision.** Use RoPE, RMSNorm (pre-norm), SwiGLU MLP, tied embeddings, no bias terms.

**Consequences.**
- Export and cross-framework comparison (E8) become feasible.
- RoPE requires explicit position offsets in the cache-aware forward — a source of bugs, so it gets its own test.
- Must get the RoPE pairing convention right for HF parity (see ARCHITECTURE §8).

---

## ADR-003 — fp16 + GradScaler for training

**Status:** Accepted

**Context.** Kaggle's T4 does not support bf16 natively. fp32 would be several times slower.

**Decision.** fp16 autocast with dynamic loss scaling; keep master weights and optimizer state in fp32.

**Consequences.**
- Must monitor for loss spikes and inf/NaN grad norms.
- Fall back to fp32 for tiny debug configs.
- If we later train on hardware with bf16, drop the scaler.

---

## ADR-004 — Train on Kaggle; use the laptop for development and inference benchmarks

**Status:** Accepted

**Context.** The laptop (Core Ultra 7 155H, Arc iGPU, 16 GB shared RAM) is far slower than a T4 for training and shares memory with the GPU. Kaggle offers a weekly free GPU quota with long sessions and a background "commit" mode.

**Decision.** All real training on Kaggle T4. Laptop for smoke tests, debugging, and the CPU/XPU/OpenVINO benchmark legs.

**Consequences.**
- Training must survive session limits → exact-resume checkpoints (FR-R2) are P0.
- Cross-device benchmark story (T4 vs iGPU vs CPU) comes almost for free.
- Weekly quota can change; verify it in the Kaggle account before planning long runs.

---

## ADR-005 — Static, preallocated KV cache with O(1) truncate

**Status:** Accepted

**Context.** Dynamic concatenation (`torch.cat` each step) is simple but reallocates every step. Speculative decoding needs cheap rollback.

**Decision.** Preallocate `[n_layers, B, n_kv_heads, max_len, head_dim]` tensors; track a single integer `length`; `truncate(n)` just resets the integer.

**Consequences.**
- No per-step allocation; compatible with CUDA graphs later.
- Wastes memory for short generations (fine at this scale: ~8 MB/sequence).
- v1 shares one `length` across the batch → equal-length prompts only.

---

## ADR-006 — Speculative decoding with a separate small draft model

**Status:** Accepted

**Context.** Options: (a) separate small draft model, (b) n-gram / prompt-lookup drafting, (c) self-speculation (early exit), (d) Medusa/EAGLE heads.

**Decision.** Start with (a): an independently trained ~4M draft on the same tokenizer and data; add distillation (P1) to raise acceptance. Consider (d) as a stretch.

**Consequences.**
- Simplest to make provably exact.
- With models this small, eager-mode launch overhead may make the draft nearly as expensive as the target (`c` close to 1). Mitigation: CUDA graphs / `torch.compile`, a shallower draft, and honest reporting.
- Requires training two models (one extra Kaggle session).

---

## ADR-007 — Write our own BPE; verify against Hugging Face `tokenizers`

**Status:** Accepted

**Context.** "Own tokenizer" is a project requirement, but correctness must be demonstrable.

**Decision.** Implement training and encoding ourselves (unique-word frequency trick for speed); use HF `tokenizers` only as a reference for compression parity and export validation.

**Consequences.**
- Pure-Python encode is slow on the full corpus → encode in chunks, cache results, or add a simple multiprocessing pool.
- Small differences from HF (merge tie-breaking) are acceptable if documented and compression is within 2%.

---

## ADR-008 — Do not depend on `torch.compile` for the iGPU path

**Status:** Accepted

**Context.** `torch.compile` on Intel XPU under Windows may require extra toolchain setup and can be fragile.

**Decision.** iGPU path runs in eager mode; OpenVINO is the "optimized" path for the iGPU. `torch.compile` and CUDA graphs are used only on CUDA.

**Consequences.** Cross-device comparisons must state which execution mode each number used (eager vs compiled vs OpenVINO).

---

## ADR-009 — Allow attention across story boundaries in v1

**Status:** Accepted (revisit if samples show cross-story leakage)

**Context.** Packing stories back-to-back in fixed windows is efficient but lets tokens attend across an EOS. A document mask fixes it but complicates the SDPA call and the cache.

**Decision.** No document mask in v1 (same as nanoGPT's default). The EOS token still teaches the model where stories end.

**Consequences.** Slight training inefficiency; negligible for benchmarking. Add a document-mask ablation only if time remains.

---

## ADR-010 — Every README number must trace to committed raw JSON

**Status:** Accepted

**Context.** Performance claims are easy to get wrong and hard to audit.

**Decision.** The benchmark harness writes one JSON per run (metrics + environment). Plots are generated by script from `results/`. The README table cites experiment IDs.

**Consequences.** Slower to write up, but every number is reproducible and defensible in an interview.

---

## Template

```markdown
## ADR-0XX — <title>

**Status:** Proposed | Accepted | Superseded by ADR-0YY

**Context.** What problem are we solving? What options exist?

**Decision.** What we chose.

**Consequences.** Trade-offs, risks, and what would make us revisit.
```
