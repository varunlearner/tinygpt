# Product Requirements Document (PRD)

**Last updated:** 2026-09-30 · **Owner:** _your name_ · **Status:** Draft v1

Priority key: **P0** = MVP, project isn't done without it · **P1** = strong version · **P2** = stretch.

## 1. Overview

`tinygpt` is a from-scratch small language model stack: tokenizer, model, training loop, and an inference engine with KV caching and speculative decoding, plus a benchmark harness that measures speedups across CUDA, CPU, and Intel XPU.

## 2. Problem statement

Existing tutorials cover training a small GPT, but rarely cover **inference optimization with correctness guarantees and honest measurement**. There is no compact, reproducible reference that (a) implements these optimizations from scratch, (b) proves they don't change model output, and (c) explains where measured speedups deviate from theory.

## 3. Users and stakeholders

| Who | Needs |
|---|---|
| Primary: the author | A project that forces deep understanding of the LLM stack |
| Reviewers / recruiters | A README that shows engineering rigor in under a minute |
| Readers who fork it | A clean, tested, reproducible codebase |

## 4. Scope

**In scope:** tokenizer, data pipeline, model, training, KV cache, speculative decoding, benchmarking, export/interop, docs, demo.
**Out of scope:** instruction tuning, RLHF, models >50M params, paid compute, production serving, multi-node training.

## 5. Functional requirements

### 5.1 Tokenizer

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-T1 | Byte-level BPE trainer, 4096 vocab (256 bytes + 2 specials + 3838 merges) | P0 | Trains on a ≤200 MB subset in <15 min on the laptop |
| FR-T2 | `encode` / `decode` | P0 | Lossless round-trip on 100% of the validation set |
| FR-T3 | Save / load tokenizer to a single JSON file | P0 | Reloaded tokenizer produces identical ids on 10k samples |
| FR-T4 | Parity check vs Hugging Face `tokenizers` trained on the same data | P1 | Compression (bytes/token) within 2% of reference; differences explained |
| FR-T5 | Export to HF `tokenizer.json` | P1 | Loads in `transformers` and matches native ids |

### 5.2 Data

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-D1 | Download TinyStories, create train/val split | P0 | Val set fixed by seed and never trained on |
| FR-D2 | Tokenize into uint16 `train.bin` / `val.bin` with EOS between stories | P0 | Total token count reported and logged |
| FR-D3 | Stream / memmap; never load full corpus into RAM | P0 | Peak RAM <4 GB during prepare on the laptop |

### 5.3 Model

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-M1 | Decoder-only GPT: RoPE, RMSNorm, SwiGLU, tied embeddings, configurable via YAML | P0 | Param count within 2% of ~28M target / ~4M draft |
| FR-M2 | Forward pass accepts an optional KV cache and start position | P0 | See FR-I3 tests |
| FR-M3 | Optional GQA (`n_kv_heads < n_heads`) | P1 | Shape + cache-size tests; ablation reported |
| FR-M4 | 42M stretch config | P2 | Trains within one Kaggle session |

### 5.4 Training

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-R1 | Training loop: fp16 + GradScaler, grad accumulation, AdamW, warmup + cosine LR, grad clipping | P0 | Can overfit a single batch to near-zero loss (sanity test) |
| FR-R2 | Checkpoint + **exact resume** (model, optimizer, scaler, RNG, data position) | P0 | Resume test: loss curve continues without discontinuity |
| FR-R3 | Log loss, LR, tokens/s, grad norm to CSV/W&B | P0 | Plots reproducible from logs |
| FR-R4 | Train target (1 epoch) and draft model | P0 | Val loss recorded; samples are grammatical stories |
| FR-R5 | Distill draft from target logits (KL loss) | P1 | Acceptance rate α improves vs independently trained draft |
| FR-R6 | DDP across Kaggle 2×T4 | P2 | **Not implemented** (single-GPU loop); out of scope for v1 |

### 5.5 Inference

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-I1 | Sampler: greedy, temperature, top-k, top-p; seeded | P0 | Deterministic given seed |
| FR-I2 | Naive generator (full recompute each step) | P0 | Baseline for E1 |
| FR-I3 | KV cache: preallocated static tensors, prefill + decode paths, O(1) `truncate` | P0 | Logits match full forward: max abs diff <1e-4 in fp32; greedy tokens identical over 1000 prompts |
| FR-I4 | Batched decode with cache (equal-length prompts in v1) | P1 | Throughput vs batch-size curve (E2) |
| FR-I5 | Speculative decoding: draft proposes k, target verifies in one pass, rejection sampling, residual resampling, bonus token | P0 | Greedy output identical to target-only greedy; sampled distribution passes statistical test |
| FR-I6 | Correct cache rollback for both models, including the "all k accepted" case | P0 | Edge-case tests pass (k=1, EOS mid-draft, max_len boundary) |
| FR-I7 | Optional: `torch.compile` / CUDA graphs for the decode step (CUDA) | P1 | E5 reports before/after (implemented as `torch.compile` default and `reduce-overhead`; CUDA-graph mode untested in the authoring sandbox) |

### 5.6 Evaluation

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-E1 | Validation loss / perplexity script | P0 | Reports nats/token and bits/byte |
| FR-E2 | Fixed prompt set (≥50) with saved generations for qualitative review | P0 | Samples committed under `results/samples/` |

### 5.7 Benchmarking

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-B1 | Config-driven harness with device abstraction (`cuda` / `xpu` / `cpu`), warmup, device sync, N runs | P0 | Writes JSON incl. hardware + software versions |
| FR-B2 | Experiments E1–E3 on ≥2 devices | P0 | Raw JSON committed; plots generated by script |
| FR-B3 | Experiments E4–E6 (theory vs measured, launch-overhead study, roofline) | P1 | Written analysis in `BENCHMARKING.md` results section |
| FR-B4 | Experiments E7–E8 (int8, cross-framework) | P2 | Results table filled |

### 5.8 Export and release

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-X1 | Export to HF `LlamaForCausalLM` layout | P1 | Logit parity vs native: max abs diff <1e-3 |
| FR-X2 | Convert to GGUF and run in llama.cpp | P2 | Generates coherent text; tokens/s recorded |
| FR-X3 | ONNX / OpenVINO export, run on CPU and iGPU | P1 | Logit parity + tokens/s recorded |
| FR-X4 | int8 weight-only quantization | P2 | Val-loss delta and speedup reported |
| FR-X5 | Publish weights + tokenizer to Hugging Face Hub; Gradio demo | P1 | Model card includes limitations |

### 5.9 Stretch

Medusa/EAGLE-style draft heads or tree speculation · paged KV cache + mini continuous-batching server · Triton decode-attention kernel. Pick at most 2–3.

## 6. Non-functional requirements

| Area | Requirement |
|---|---|
| Reproducibility | Pinned `requirements.txt`; seeds everywhere; YAML configs; env report saved with every result |
| Resource limits | Laptop workflows fit in 16 GB RAM; training resumes across Kaggle session limits |
| Measurement rigor | ≥3 warmup + ≥10 timed runs; median + IQR; device sync; thermal notes for laptop runs |
| Code quality | Type hints, `ruff`, `pytest`, CI running CPU tests on every push |
| Documentation | Every doc in `docs/` stays current; ADR added for each significant decision |
| Honesty | No number in the README without an experiment ID and committed JSON |

## 7. Deliverables

1. Repo with code, tests, configs, CI.
2. Trained weights (target, draft) + tokenizer on HF Hub.
3. `results/` with raw JSON for every experiment.
4. README with results table and plots.
5. (Optional) Blog post walking through the KV-cache and speculative-decoding results.

## 8. Milestones

See [`ROADMAP.md`](ROADMAP.md). Summary: tokenizer → model + training → full training → KV cache → speculative decoding → devices/export → write-up.

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle session dies mid-training | High | Medium | Checkpoint every ~15 min or 500 steps; exact-resume tested early |
| fp16 loss spikes / NaNs | Medium | Medium | GradScaler, grad clipping, monitor grad norm; fp32 fallback for debugging |
| Speculative decoding shows no speedup (launch overhead dominates tiny models) | High | Medium | CUDA graphs / `torch.compile`; deeper target vs shallower draft; report honestly with analysis |
| Intel XPU / driver problems on Windows | Medium | Low | CPU fallback; OpenVINO path; keep iGPU work non-blocking |
| 16 GB RAM pressure (iGPU shares it) | Medium | Medium | memmap everything; small subsets for tokenizer training; close other apps |
| RoPE convention mismatch breaks HF/GGUF export | Medium | Low | Parity test early; document the rotate-half vs interleaved convention |
| Scope creep | High | High | Tiers + explicit cut line in `ROADMAP.md` |
| Laptop thermal throttling skews benchmarks | Medium | Medium | Plugged in, performance mode, cooldown between runs, record clocks/temps if possible |

## 10. Dependencies

PyTorch (CUDA build on Kaggle, XPU build on laptop) · `regex` · `numpy` · `pyyaml` · `pytest` · `matplotlib` · Hugging Face `tokenizers` and `transformers` (reference/parity only) · OpenVINO (P1) · llama.cpp (P2).

## 11. Open questions

| # | Question | Default until decided |
|---|---|---|
| 1 | MHA or GQA as the default target? | MHA default; GQA as ablation |
| 2 | Context length 512 or 256? | 512 |
| 3 | Vocab 4096 or 8192? | 4096 (see ADR-001) |
| 4 | Publish weights publicly? | Yes, after checking dataset license terms |
| 5 | Batched decode with ragged prompts? | Out of scope for v1 (equal-length only) |

## 12. Definition of done

All P0 requirements met · CI green · README results table filled from committed JSON · at least one P1 analysis (E4 or E6) written up · docs updated · weights released.
