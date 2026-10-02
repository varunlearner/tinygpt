# Topics and Reading List

**Last updated:** 2026-09-30

How to use this: study a topic *just before* the phase that needs it (see the "Needed in" column). For each topic, answer the **checkpoint question** in your own words before moving on — if you can't, you don't understand it yet. Log the answers in `LEARNINGS.md`.

Tags: **[Core]** needed for the MVP · **[Should]** needed for the strong version · **[Stretch]** optional depth.

---

## 1. Tokenization

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Byte-pair encoding: merge algorithm and why it works | Core | Phase 1 | Why does BPE start from bytes, and what guarantees it can encode any string? |
| Byte-level BPE and pre-tokenization regex | Core | Phase 1 | What breaks if merges may cross word or space boundaries? |
| Vocab-size trade-offs (embedding params, sequence length, softmax cost) | Core | Phase 1 | Why is 4096 reasonable for TinyStories but poor for multilingual web text? |
| Compression metrics (bytes/token, tokens/story) | Should | Phase 1 | How do you compare two tokenizers fairly? |
| Special tokens and document separators | Core | Phase 1 | Why does the model need an EOS token if we pack stories together? |

## 2. Transformer architecture

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Scaled dot-product attention, causal masking | Core | Phase 2 | Why divide by √d_k? What does the mask do to the softmax? |
| Multi-head attention and head dimension | Core | Phase 2 | What does splitting into heads change computationally? |
| Residual stream, pre-norm vs post-norm | Core | Phase 2 | Why is pre-norm easier to train at depth? |
| RMSNorm vs LayerNorm | Core | Phase 2 | What does RMSNorm drop, and why does it still work? |
| SwiGLU MLP | Core | Phase 2 | Why is the hidden size ~8/3·d instead of 4·d? |
| Rotary position embeddings (RoPE) | Core | Phase 2 / 4 | How does RoPE encode *relative* position through a dot product? Why does the cache need `start_pos`? |
| Weight tying | Core | Phase 2 | What is tied, and why does it help small models? |
| Multi-query / grouped-query attention | Should | Phase 4 | How does GQA shrink the KV cache and what does it cost in quality? |

## 3. Training

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Cross-entropy, perplexity, bits-per-byte | Core | Phase 2 | Why is bits-per-byte a fairer metric across tokenizers? |
| AdamW and decoupled weight decay; which params to exclude | Core | Phase 2 | Why not decay norm weights and embeddings? |
| Warmup + cosine schedule | Core | Phase 2 | What goes wrong without warmup? |
| Gradient clipping and accumulation | Core | Phase 2 | Are accumulated micro-batches equivalent to a big batch? What must you scale? |
| Mixed precision: fp16 vs bf16, loss scaling | Core | Phase 2 | Why does fp16 need a loss scaler but bf16 doesn't? |
| Scaling laws / Chinchilla-optimal tokens | Should | Phase 3 | Is one epoch over ~0.5B tokens compute-optimal for ~28M params? |
| Efficient data loading with memmap | Core | Phase 1–2 | Why random windows instead of shuffled documents? |
| Checkpoint/resume, RNG state | Core | Phase 2 | What state must be saved for a truly exact resume? |
| Data-parallel training (DDP) | Stretch | Phase 6 | What is all-reduced, and when? |

## 4. Inference fundamentals

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Autoregressive decoding; prefill vs decode | Core | Phase 4 | Which phase is compute-bound and which is memory-bound, and why? |
| Sampling: greedy, temperature, top-k, top-p | Core | Phase 4 | How does temperature change acceptance in speculative decoding? |
| Arithmetic intensity and the roofline model | Should | Phase 4 / 6 | Where does batch-1 decode sit on the roofline, and what moves it? |
| Memory-bandwidth utilization (MBU) | Should | Phase 6 | How do you compute an upper bound on tokens/s from bandwidth? |

## 5. KV cache

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Why caching K and V (not Q) is enough | Core | Phase 4 | What exactly is recomputed without a cache, and why is it wasted? |
| Cache memory formula | Core | Phase 4 | Compute the cache size for this model at batch 32, ctx 512, fp16. |
| Static vs dynamic allocation | Core | Phase 4 | Why is preallocation friendly to CUDA graphs? |
| Rollback / truncation | Core | Phase 5 | Why is rollback O(1) with a length pointer? |
| Paged attention (vLLM) | Stretch | Phase 6 | What fragmentation problem does paging solve? |

## 6. Speculative decoding

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Rejection sampling and why the output distribution is exact | Core | Phase 5 | Derive why accept-with-min(1,p/q) plus residual resampling yields samples from p. |
| Acceptance rate α and expected tokens per round | Core | Phase 5 | Derive `(1 − α^(k+1)) / (1 − α)`. |
| Cost coefficient c and the speedup formula | Core | Phase 5 | What happens to the speedup as c → 1? |
| Choosing k and the draft model | Core | Phase 5 | Why is there an optimal k rather than "bigger is better"? |
| Draft distillation (KL to target logits) | Should | Phase 5 | Why does distillation raise α? |
| Cache handling for two models | Core | Phase 5 | Why is the draft cache one entry short after all k are accepted? |
| Tree speculation, Medusa, EAGLE, self-speculation | Stretch | Phase 6 | How do drafting heads avoid a separate model? |

## 7. Performance engineering

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| GPU execution model: kernels, streams, async launch | Should | Phase 4 | Why is a wall-clock timer meaningless without a device sync? |
| Kernel-launch overhead and CUDA graphs | Should | Phase 5 | Why does a 28M model on a T4 not run at the hardware's speed limit? |
| `torch.compile` basics | Should | Phase 5 | What does it fuse and what does it not help with in decode? |
| SDPA backends (flash / mem-efficient / math) | Should | Phase 2 | Why is FlashAttention-2 unavailable on a T4? |
| Profiling with `torch.profiler` | Should | Phase 5 | How do you recognize a launch-bound workload in a trace? |
| Benchmark hygiene: warmup, sync, variance, thermals | Core | Phase 4 | What can make the same benchmark differ by 30% on a laptop? |
| Triton kernels | Stretch | Phase 6 | What would a fused decode-attention kernel save? |

## 8. Quantization

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Weight-only int8 (per-channel, symmetric) | Stretch | Phase 6 | Why does weight-only quantization speed up a memory-bound decode? |
| Quality measurement after quantization | Stretch | Phase 6 | Why measure val-loss delta rather than eyeballing samples? |

## 9. Deployment and interoperability

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| HF `LlamaForCausalLM` layout and config | Should | Phase 6 | Which of your tensors map to which HF names? |
| RoPE pairing conventions (interleaved vs rotate-half) | Should | Phase 6 | Why would a mismatch yield plausible-looking but wrong logits? |
| GGUF and llama.cpp | Stretch | Phase 6 | What does GGUF store beyond weights? |
| ONNX and OpenVINO; CPU / iGPU / NPU targets | Should | Phase 6 | Why is the NPU a poor fit for a growing KV cache? |
| Intel XPU (SYCL) in PyTorch | Should | Phase 0 / 6 | What differs from CUDA in sync and memory behavior? |

## 10. Evaluation

| Topic | Tag | Needed in | Checkpoint question |
|---|---|---|---|
| Validation loss / perplexity | Core | Phase 3 | Why is perplexity across different vocabularies not comparable? |
| Qualitative sampling protocol | Core | Phase 3 | How do you avoid cherry-picking samples? |
| Model-graded story quality (grammar, consistency, creativity — as in the TinyStories paper) | Stretch | Phase 7 | What are the biases of model-graded evaluation? |

## 11. Engineering practice

| Topic | Tag | Checkpoint question |
|---|---|---|
| Testing numerical code (tolerances, seeds, statistical tests) | Core | Why chi-square instead of exact equality for sampling tests? |
| Config-driven experiments; reproducibility | Core | What would someone need to reproduce your headline number? |
| Architecture decision records | Should | Which decision would be hardest to explain in an interview without an ADR? |
| CI for ML repos (CPU-only test subset) | Should | Which tests can run on CPU in <2 minutes? |

---

## Reading list

References are given by author, year, and title so you can search for them; check publication details yourself.

### Papers

**Data and model**
- Eldan & Li (2023) — *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Zhang & Sennrich (2019) — *Root Mean Square Layer Normalization*
- Shazeer (2020) — *GLU Variants Improve Transformer*
- Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Touvron et al. (2023) — *LLaMA: Open and Efficient Foundation Language Models*

**Tokenization**
- Sennrich, Haddow & Birch (2016) — *Neural Machine Translation of Rare Words with Subword Units*
- Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2; byte-level BPE)

**Training**
- Loshchilov & Hutter (2019) — *Decoupled Weight Decay Regularization* (AdamW)
- Micikevicius et al. (2018) — *Mixed Precision Training*
- Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models* (Chinchilla)

**Inference optimization**
- Leviathan, Kalman & Matias (2022) — *Fast Inference from Transformers via Speculative Decoding*
- Chen et al. (2023) — *Accelerating Large Language Model Decoding with Speculative Sampling*
- Hinton, Vinyals & Dean (2015) — *Distilling the Knowledge in a Neural Network*
- Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Kwon et al. (2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- Cai et al. (2024) — *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
- Li et al. (2024) — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
- Dettmers et al. (2022) — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- Williams, Waterman & Patterson (2009) — *Roofline: An Insightful Visual Performance Model for Multicore Architectures*

### Code to read (in this order)

1. **nanoGPT** (Karpathy) — the cleanest minimal training loop and model.
2. **llama2.c** (Karpathy) — a tiny Llama-style model trained on TinyStories; a direct reference point for your model and for small vocab choices.
3. **gpt-fast** (PyTorch team) — minimal, fast decoding with a static KV cache, `torch.compile`, quantization, and speculative decoding.
4. **llama.cpp** — how a production runtime lays out GGUF, KV cache, and sampling.
5. **Hugging Face `transformers`** — `LlamaModel` (for export parity) and assisted generation (for comparison with your speculative implementation).

### Videos

- Karpathy — *Let's build GPT: from scratch, in code, spelled out*
- Karpathy — *Let's build the GPT Tokenizer*

## Suggested study order

| Week | Topics |
|---|---|
| Before Phase 1 | §1 (BPE), Sennrich et al., Karpathy tokenizer video |
| Before Phase 2 | §2, §3, nanoGPT, RoPE paper, mixed precision |
| Before Phase 4 | §4, §5, arithmetic intensity, benchmarking hygiene |
| Before Phase 5 | §6, both speculative decoding papers (derive the formulas by hand), gpt-fast |
| Before Phase 6 | §7–§9, roofline paper, HF Llama code, OpenVINO docs |
