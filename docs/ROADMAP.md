# Roadmap

**Last updated:** 2026-09-30 · **Total estimate:** ~3 weeks part-time (~22 days of effort)

Each phase has **tasks**, an **exit criterion** you can check objectively, and the **PRD IDs** it covers. Don't start the next phase until the exit criterion is met — especially for correctness tests.

## Implementation status (2026-09-30)

| Phase | Code | Tested in sandbox (CPU) | Left for you |
|---|---|---|---|
| 0 Setup | done | yes | install PyTorch (XPU build on the laptop), run `probe_xpu.py`, record results |
| 1 Tokenizer + data | done | yes (unit tests + toy corpus) | download real data, run tokenizer + prepare, record token counts |
| 2 Model + training loop | done | yes (incl. exact resume, distillation) | LR sweep on the real config (optional) |
| 3 Full training | done | real 28M config takes steps; `--compile` works | **run the training on Kaggle** |
| 4 KV cache | done | yes (equivalence tests) | run E1, E2 on GPU + laptop |
| 5 Speculative decoding | done | yes (greedy identity + statistical distribution test) | run E3-E5 |
| 6 Devices / export | done | HF export + int8 tested; OpenVINO/llama.cpp/XPU not | run on your iGPU, OpenVINO, llama.cpp |
| 7 Write-up | docs written | - | fill README numbers from `results/SUMMARY.md` |

Step-by-step commands: [`RUNBOOK.md`](RUNBOOK.md).

## Overview

| # | Phase | Est. | Where it runs | Covers |
|---|---|---|---|---|
| 0 | Setup | 0.5–1 d | Laptop | — |
| 1 | Tokenizer + data | 2–3 d | Laptop | FR-T*, FR-D* |
| 2 | Model + training loop | 3 d | Laptop (smoke) → Kaggle | FR-M*, FR-R1–R3 |
| 3 | Full training | 1–2 Kaggle sessions | Kaggle T4 | FR-R4, FR-E* |
| 4 | KV cache | 3–4 d | Laptop + Kaggle | FR-I1–I4 |
| 5 | Speculative decoding | 3–4 d | Laptop + Kaggle | FR-I5–I7 |
| 6 | Devices, quantization, export | 2–3 d | Laptop + Kaggle | FR-B2–B4, FR-X* |
| 7 | Write-up + release | 2 d | Anywhere | FR-X5, README |

---

## Phase 0 — Setup (0.5–1 day)

- [ ] Create repo, `pyproject.toml`/`requirements.txt`, `ruff`, `pytest`, CI skeleton (CPU tests)
- [ ] Set up laptop env with XPU-enabled PyTorch; run `scripts/probe_xpu.py`; record TFLOPS and bandwidth in `ENVIRONMENT.md`
- [ ] Create Kaggle account settings: phone verification, check GPU quota, enable internet for notebooks
- [ ] Download TinyStories; carve out a 100–200 MB subset for tokenizer training
- [ ] Write `scripts/env_report.py` (device, driver, versions → JSON)

**Exit:** `pytest` runs green on an empty suite in CI; probe results recorded.

## Phase 1 — Tokenizer + data (2–3 days)

- [ ] Pre-tokenization regex + word-frequency table
- [ ] BPE merge training (unique-word trick); 3838 merges
- [ ] `encode` / `decode` / `save` / `load`; special tokens
- [ ] Tests: round-trip on val set, determinism, special-token handling
- [ ] Parity check vs HF `tokenizers` (bytes/token within 2%)
- [ ] `data/prepare.py`: chunked encode → `train.bin` / `val.bin` (uint16), print token counts
- [ ] Peak RAM check during prepare (<4 GB)

**Exit:** FR-T1–T3, FR-D1–D3 acceptance criteria met; token count recorded in README.

## Phase 2 — Model + training loop (3 days)

- [ ] `model/gpt.py`: RMSNorm, RoPE, attention with SDPA, SwiGLU, tied head; YAML configs
- [ ] Unit tests: output shapes, causal-mask test (changing a future token doesn't change earlier logits), param counts
- [ ] Training loop: AdamW param groups, warmup + cosine, grad accumulation, fp16 + GradScaler, clipping
- [ ] Checkpoint + **exact resume** test
- [ ] Sanity: overfit one batch to near-zero loss
- [ ] Smoke test on laptop iGPU/CPU with a tiny config (a few hundred steps)
- [ ] Small LR sweep on a tiny config; measure real T4 tokens/s on Kaggle for 10 minutes; update ARCHITECTURE §5

**Exit:** overfit test passes, resume test passes, T4 throughput measured.

## Phase 3 — Full training (1–2 Kaggle sessions)

- [ ] Upload tokenized shards as a Kaggle Dataset; pull code from GitHub in the notebook
- [ ] Train target (`target_28m`), 1 epoch, in "Save & Run All (Commit)" background mode; watch first 30 min for spikes
- [ ] Train draft (`draft_4m`) same data
- [ ] Evaluate: val loss, bits/byte, 50 saved sample generations
- [ ] Save weights + tokenizer as a Kaggle Dataset and to local disk

**Exit:** both models trained; val loss recorded; samples are grammatical stories; weights backed up in two places.

## Phase 4 — KV cache (3–4 days)

- [ ] Sampler (greedy / temperature / top-k / top-p) with seeded RNG
- [ ] Naive generator (baseline)
- [ ] `KVCache` (static, `update` / `advance` / `truncate`); cache-aware forward with `start_pos`
- [ ] **Tests:** prefill + decode logits equal full forward (fp32, atol 1e-4); greedy tokens identical over 1000 prompts; truncate then re-decode reproduces same logits
- [ ] Benchmark harness v1 (warmup, sync, N runs, JSON with env block)
- [ ] Run **E1** (naive vs cache vs length) and **E2** (batch scaling) on T4 + laptop
- [ ] Write predictions in `LEARNINGS.md` *before* running; record gaps after

**Exit:** all cache tests pass; E1 and E2 JSON committed for ≥2 devices.

## Phase 5 — Speculative decoding (3–4 days)

- [ ] Implement propose → verify → accept/reject → residual resample → bonus token
- [ ] Two caches with correct rollback, including the all-k-accepted case
- [ ] **Tests:** greedy identity vs target-only greedy; toy-model statistical distribution test (vocab 8, ≥200k samples); edge cases (k=1, EOS in draft block, max_len boundary)
- [ ] Run **E3** (k × temperature × draft size) on T4 + laptop
- [ ] Run **E4** (fit α and c, compare with formula) and **E5** (eager vs `torch.compile` / CUDA graphs)
- [ ] (P1) Distill the draft from the target's logits; re-run E3 with the distilled draft

**Exit:** exactness tests pass; E3 JSON committed; E4 analysis written.

## Phase 6 — Devices, quantization, export (2–3 days)

- [ ] Run E1–E3 on the iGPU (XPU, eager) and CPU; compute **E6** roofline / memory-bandwidth utilization
- [ ] Export to HF `LlamaForCausalLM` with logit-parity test (P1)
- [ ] ONNX → OpenVINO on CPU and iGPU; parity + tokens/s (P1)
- [ ] int8 weight-only quantization; val-loss delta + speedup (**E7**, P2)
- [ ] GGUF → llama.cpp comparison (**E8**, P2)
- [ ] Pick 0–2 stretch items (Medusa heads / paged KV cache / Triton kernel) **only if** phases 0–5 are complete

**Exit:** at least the P1 export items done; E6 written up.

## Phase 7 — Write-up + release (2 days)

- [ ] Fill README results table from committed JSON; generate plots via `bench.plot`
- [ ] Model card (intended use, limitations, training data, license note)
- [ ] Publish weights + tokenizer to Hugging Face Hub
- [ ] Gradio demo (naive vs cache vs speculative, live tokens/s)
- [ ] (Optional) blog post: "Why my speculative decoding speedup was X, not Y"
- [ ] Update `LEARNINGS.md` interview prep; tick every item in the PRD definition of done

**Exit:** a stranger can clone the repo and reproduce the headline numbers.

---

## Cut line (if time runs short)

**Drop in this order:** DDP → Triton kernel → paged KV cache / server → Medusa heads → GGUF/llama.cpp → int8 → OpenVINO → 42M stretch → distillation.

**Never drop:** correctness tests, E1–E3 on ≥2 devices, README results table with raw JSON, honest analysis of any surprising result.

## Weekly check-in template

```markdown
### Week of YYYY-MM-DD
- Phase:
- Done this week:
- Blocked on:
- Predictions made / results in:
- Ahead / on track / behind (cut something?):
- Next week's single most important task:
```

## Milestone calendar (suggested)

| Week | Goal |
|---|---|
| 1 | Phases 0–2 done; first Kaggle run started by the weekend |
| 2 | Phase 3 done; Phases 4–5 in progress; E1–E3 on T4 |
| 3 | Phases 5–7: laptop benchmarks, exports, write-up, release |
