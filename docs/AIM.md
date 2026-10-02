# Aim

**Last updated:** 2026-09-30 · **Owner:** _your name_

## One-sentence aim

Train a small GPT from scratch, then make its inference measurably faster with KV caching and speculative decoding — proving the optimizations are *correct* and reporting the speedups *honestly*, across three kinds of hardware.

## Motivation

- **Training is a one-time cost; inference is forever.** Serving cost and latency are where LLM engineering time goes in practice.
- **Autoregressive decoding is memory-bound.** Each generated token re-reads all the model weights. KV caching and speculative decoding both attack this, from different angles.
- **Small scale makes everything inspectable.** With a ~28M-parameter model you can read every tensor, unit-test every claim, and run the full experiment matrix on free hardware.
- **Owning the whole stack builds real understanding.** Tokenizer → model → training → inference → benchmarks. No black boxes.

## Objectives

| ID | Objective |
|---|---|
| O1 | Build a byte-level BPE tokenizer from scratch and verify it against a reference implementation |
| O2 | Train a ~28M-parameter Llama-style GPT (and a ~4M draft model) on TinyStories using only free compute |
| O3 | Implement KV caching; prove equivalence with the no-cache path; measure speedup vs sequence length and batch size |
| O4 | Implement speculative decoding with correct rejection sampling; prove it preserves the target distribution; measure speedup vs k, temperature, and draft size |
| O5 | Compare theory to measurement (speedup formula, roofline / memory-bandwidth utilization) and explain any gap |
| O6 | Run the same benchmarks on an NVIDIA T4, a CPU, and an Intel Arc iGPU |
| O7 | Package it so a stranger can reproduce the headline numbers from the README |

## Success criteria

The project is a success when **all P0 criteria** hold:

1. **Correctness is proven, not asserted.** KV-cache logits match full-forward logits; speculative decoding under greedy sampling is token-identical to plain greedy decoding; a statistical test confirms sampled distributions match. All tests pass in CI.
2. **Speedups are measured with rigorous method.** Warmup, device synchronization, ≥10 runs, median and spread reported, hardware and versions recorded next to every number.
3. **The README carries one results table and 3–5 plots**, each traceable to raw JSON committed in `results/`.
4. **Negative and surprising results are reported and explained**, not hidden. (A measured 1.1× with a clear diagnosis is a better result than an unexplained 3×.)
5. **Reproducible:** pinned dependencies, seeds, YAML configs, one command per stage.

## What a reviewer should conclude in 60 seconds

> "This person understands the whole LLM pipeline, writes tests before making performance claims, measures carefully, and can explain *why* the numbers are what they are."

## Non-goals

- Competing with vLLM, llama.cpp, or TensorRT-LLM on performance.
- Training a general-purpose or instruction-following model.
- Training anything above ~50M parameters or on paid compute.
- Training on the laptop iGPU (it is for development and inference benchmarking).
- A production serving system (a mini continuous-batching server is a stretch item, not a goal).
- Claiming results on hardware not actually measured.

## Constraints

| Constraint | Detail |
|---|---|
| Compute | Free tiers only — Kaggle T4 (~30 GPU-hours/week, verify in your account) for training |
| Laptop | Core Ultra 7 155H, 16 GB RAM (15.4 usable) shared with the iGPU |
| Time | ~3 weeks part-time (see `ROADMAP.md`) |
| Precision | T4 has no bf16 → fp16 with loss scaling |

## Guiding principles

1. **Correctness before speed.** Never benchmark code that hasn't passed its equivalence test.
2. **Measure, don't guess.** Every claim gets an experiment ID and a JSON file.
3. **Write predictions down before running experiments** (see `LEARNINGS.md`), then explain the gaps.
4. **Small, reviewable increments.** Each phase ends with an exit criterion that is checkable.
5. **Honest scope.** Say what it is and isn't.

## Draft resume bullets (fill in the X's only with measured numbers)

- Trained a 28M-parameter GPT from scratch on TinyStories using a custom byte-level BPE tokenizer (val loss X) on free-tier T4 GPUs.
- Implemented KV caching (X× faster at 480 generated tokens) and speculative decoding with a distilled 4M draft model (X× speedup at Y% acceptance), verified output-distribution-preserving via statistical tests.
- Benchmarked across CUDA, CPU, and Intel Arc iGPU with roofline analysis; exported to HF Llama / GGUF / OpenVINO with logit-parity tests.
