# Learnings

**Last updated:** 2026-09-30

This is a working notebook, not a polished document. Its purpose is to make you *predict* before you measure, *explain* after you measure, and turn the whole project into interview-ready understanding.

## 1. Learning goals

Tick a box only when you could explain it to someone else without notes.

### Fundamentals
- [ ] I can implement BPE training and encoding without looking at a reference
- [ ] I can explain every tensor shape in one transformer block, forward and backward
- [ ] I can explain why RoPE needs absolute positions during cached decoding
- [ ] I can explain fp16 loss scaling and what happens without it

### Inference engineering
- [ ] I can derive the KV-cache memory formula and compute it for any config
- [ ] I can explain prefill vs decode and which is compute- vs memory-bound
- [ ] I can prove (informally) that speculative decoding preserves the target distribution
- [ ] I can derive the expected-tokens-per-round and speedup formulas
- [ ] I can read a profiler trace and say whether a workload is launch-bound or bandwidth-bound

### Measurement and engineering practice
- [ ] I can list five ways a GPU benchmark can lie and how I guarded against each
- [ ] I can design a statistical test for a sampling algorithm
- [ ] I can explain the gap between theoretical and measured speedups for *my* numbers

## 2. Predictions vs reality

**Rule:** fill in the *Prediction* column and the date **before** running the experiment. Fill in the rest afterward. The interesting part is the "Why the gap" column.

| Date | Experiment | Prediction (before) | Actual (after) | Why the gap? |
|---|---|---|---|---|
| | E1: cache speedup at 480 new tokens (T4) | | | |
| | E1: cache speedup at 32 new tokens (T4) | | | |
| | E2: batch size where throughput stops scaling (iGPU) | | | |
| | E3: best speculative speedup (T4, greedy) | | | |
| | E3: acceptance rate at temperature 1.0 vs 0 | | | |
| | E4: does the formula over- or under-predict? | | | |
| | E5: gain from CUDA graphs | | | |
| | E6: MBU at batch 1 (iGPU) | | | |
| | Distilled vs independent draft: change in α | | | |
| | Val loss of the 28M model after 1 epoch | | | |

## 3. Open questions to answer by building

1. Is batch-1 decode on this model launch-bound? What does the profiler trace look like?
2. How large must the target be relative to the draft before speculative decoding pays off on each device?
3. Does the iGPU benefit *more* from speculative decoding than the T4 (because it is more bandwidth-starved)?
4. How much does GQA shrink the cache, and does it change the speculative-decoding tradeoff?
5. At what temperature does speculative decoding stop being worth it?
6. How far is a custom implementation from HF `generate` and llama.cpp on the same hardware?

## 4. Log

Newest entries on top. Use the template below.

```markdown
### YYYY-MM-DD — <short title>
**Phase:** N · **Time spent:** h
**What I did:**
**What surprised me:**
**Bug of the day (symptom → root cause → fix):**
**What I now understand that I didn't yesterday:**
**Open questions:**
**Kaggle quota remaining (if relevant):**
```

### 2026-09-30 — Project planning
**Phase:** 0 · **What I did:** Wrote the design docs (aim, PRD, architecture, decisions, roadmap, benchmarking, environment, topics).
**What I now understand:** The value of the project is in the method — prove correctness, measure, explain gaps — more than in any single speedup number.
**Open questions:** See §3.

## 5. Bug museum

Keep a list of bugs that cost >30 minutes. They are the best interview stories.

| Date | Symptom | Root cause | Fix | Test that now prevents it |
|---|---|---|---|---|
| | | | | |

## 6. Interview prep — questions you should be able to answer

Answer each in 2–4 sentences, out loud, using *your* numbers.

**Tokenizer and data**
1. Why byte-level BPE? What does the pre-tokenization regex do?
2. Why 4096 tokens? What would change with 32k?
3. How did you verify your tokenizer is correct?

**Model and training**
4. Why RoPE, RMSNorm, and SwiGLU? What would you lose with GPT-2 style?
5. What did you do about fp16 instability on a T4?
6. How did you handle Kaggle's session limits? How did you test resume?
7. Is one epoch compute-optimal for your model size? Why or why not?

**KV cache**
8. What is cached, why, and how large is it for your model?
9. How did you prove your cache doesn't change outputs?
10. Why does the speedup depend on generation length?

**Speculative decoding**
11. Explain the algorithm end to end. Why is the output distribution unchanged?
12. What are α and c, and how did you measure them?
13. Why did speculative decoding give X× and not the theoretical Y×?
14. What breaks if the draft and target use different sampling processing?
15. What is the subtle cache bug when all k tokens are accepted?

**Systems and measurement**
16. Is decode compute-bound or memory-bound here? How do you know?
17. How did you make your benchmarks trustworthy on a laptop?
18. Your iGPU vs T4 numbers — what explains the difference?
19. What would you do next with another month? With paid GPUs?

**Meta**
20. What was the hardest bug, and how did you find it?
21. What is the biggest limitation of this project, honestly?

## 7. Things I would do next (parking lot)

- Ragged-batch decode with per-sequence cache lengths
- Tree-based speculation
- Document-masked packing (ADR-009 ablation)
- Larger target (100M+) on rented GPUs to see how speedups change with scale
