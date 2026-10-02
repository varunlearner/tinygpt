# tinygpt

**A ~28M-parameter GPT trained from scratch on TinyStories, with a from-scratch BPE tokenizer, KV caching, and speculative decoding — benchmarked across an NVIDIA T4, a CPU, and an Intel Arc iGPU.**

| | |
|---|---|
| Status | **Code complete and tested on CPU - what remains is running the training** (see [`docs/RUNBOOK.md`](docs/RUNBOOK.md)) |
| Last updated | 2026-09-30 |
| Compute | Kaggle T4 (training) · Core Ultra 7 155H laptop (development + inference benchmarks) |
| Working name | `tinygpt` - rename freely |

---

## Why this project exists

Most "I built a GPT" projects stop at training. Production LLM work is mostly about **serving**: making token generation fast without changing its output. This project builds the whole stack at a scale where every layer can be understood, tested, and measured on free hardware:

1. **Own the data path** — a byte-level BPE tokenizer written from scratch and verified against a reference implementation.
2. **Own the model** — a Llama-style decoder (RoPE, RMSNorm, SwiGLU) trained from scratch.
3. **Own the inference path** — KV caching and speculative decoding implemented by hand, proven correct by tests, and measured honestly.

See [`docs/AIM.md`](docs/AIM.md) for the full motivation and success criteria.

## Headline results

> **No results yet.** Do not fill this table until the matching experiment in [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) has been run and its raw JSON is committed under `results/`.

| Metric | Result | Experiment |
|---|---|---|
| Target model | ~28M params · val loss _TBD_ | — |
| Tokenizer | 4096 vocab · _TBD_ bytes/token (HF reference: _TBD_) | — |
| KV cache vs naive, 480 new tokens, T4 | _TBD_× | E1 |
| Speculative decoding, best config, T4 | _TBD_× at α = _TBD_, k = _TBD_ | E3 |
| Speculative decoding, iGPU | _TBD_× | E3 |
| Theory vs measured speedup gap | _TBD_ | E4 |
| int8 weight-only: speedup / val-loss delta | _TBD_ / _TBD_ | E7 |

## What's in the box

| Component | What it does | Code | Tests |
|---|---|---|---|
| Tokenizer | Byte-level BPE trainer + encoder/decoder, 4096 vocab, HF `tokenizer.json` export | `tinygpt/tokenizer/`, `export/tokenizer_to_hf.py` | `test_tokenizer.py`, `test_export.py` |
| Data | TinyStories download -> uint16 memmap shards (multiprocess, bounded RAM) | `tinygpt/data/` | `test_data.py` |
| Model | Llama-style GPT (RoPE, RMSNorm, SwiGLU, optional GQA), cache-aware forward | `tinygpt/model/` | `test_model.py` |
| Training | fp16/bf16, grad accumulation, cosine LR, exact resume, distillation, time limit | `tinygpt/train.py` | `test_train.py` |
| Inference | sampling, naive generate, **KV cache**, **speculative decoding** | `tinygpt/infer/` | `test_kv_cache.py`, `test_speculative.py` |
| Benchmarks | E1 (cache), E2 (batch), E3 (speculative), E5 (launch overhead), E4/E6 analysis, plots | `bench/` | smoke test |
| Export | HF Llama (logit parity), int8 weight-only, OpenVINO, external-engine comparison | `export/` | `test_export.py` |
| Kaggle | CPU data-prep notebook + GPU train/benchmark notebook | `notebooks/` | - |

Not implemented (P2 in the PRD): multi-GPU DDP, Medusa/EAGLE heads, paged KV cache, Triton kernels.

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .     # + the right PyTorch build, see docs/RUNBOOK.md
python -m pytest                                        # unit + equivalence + statistical tests
python scripts/smoke_test.py                            # whole pipeline on a tiny offline toy corpus (~1-3 min)

# real pipeline (details and Kaggle notebooks: docs/RUNBOOK.md)
python -m tinygpt.data.download --out data/raw
python -m tinygpt.tokenizer.train --input data/raw/TinyStoriesV2-GPT4-train.txt --max-bytes 200000000 --vocab-size 4096 --out data/tok/tokenizer.json
python -m tinygpt.data.prepare --tok data/tok/tokenizer.json --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt --out data/shards --workers 4

python -m tinygpt.train --config configs/target_28m.yaml --compile            # <- the long step (resumable with --resume)
python -m tinygpt.train --config configs/draft_4m.yaml --compile
python -m tinygpt.train --config configs/draft_4m_distilled.yaml --compile

python -m tinygpt.infer.cli --ckpt runs/target_28m/model.pt --draft runs/draft_4m_distilled/model.pt --mode speculative --k 4
python -m bench.run E1 --target runs/target_28m/model.pt                       # then E2, E3, E5; see RUNBOOK
python -m bench.analyze && python -m bench.plot
```

## Repository layout

```
tinygpt/
|-- README.md  pyproject.toml  requirements.txt  Makefile
|-- docs/                        # design docs + RUNBOOK.md (start there)
|-- configs/                     # target_28m, target_28m_gqa, target_42m, draft_4m, draft_2l, draft_4m_distilled, tiny_*
|-- tinygpt/
|   |-- tokenizer/  bpe.py  train.py
|   |-- data/       download.py  prepare.py  loader.py
|   |-- model/      config.py  gpt.py  kv_cache.py  io.py
|   |-- infer/      sampling.py  generate.py  speculative.py  cli.py
|   |-- eval/       perplexity.py  samples.py
|   |-- train.py  utils.py
|-- bench/          run.py  analyze.py  plot.py  common.py
|-- export/         to_hf_llama.py  tokenizer_to_hf.py  quantize.py  to_openvino.py  bench_external.py
|-- scripts/        smoke_test.py  probe_xpu.py  env_report.py  tokenizer_parity.py  lr_sweep.py  make_toy_corpus.py
|-- notebooks/      01_prepare_data_cpu.ipynb  02_train_and_benchmark_gpu.ipynb
|-- tests/          conftest.py  test_tokenizer|model|kv_cache|speculative|data|train|export.py
|-- results/        raw benchmark JSON, plots, samples (committed)
```

## Documentation index

| Doc | Read it to learn... |
|---|---|
| [`AIM.md`](docs/AIM.md) | Why this project exists, what "done" means, what's out of scope |
| [`PRD.md`](docs/PRD.md) | Requirements with IDs, priorities, and acceptance criteria |
| [`RUNBOOK.md`](docs/RUNBOOK.md) | **Every command in order** - the file to follow |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, model configs, KV cache and speculative decoding design |
| [`DECISIONS.md`](docs/DECISIONS.md) | Architecture decision records (the "why" behind each choice) |
| [`ROADMAP.md`](docs/ROADMAP.md) | Phases, exit criteria, time estimates, cut line |
| [`BENCHMARKING.md`](docs/BENCHMARKING.md) | Methodology, experiment matrix, correctness tests |
| [`ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Kaggle + laptop setup, RAM discipline, reproducibility |
| [`TOPICS.md`](docs/TOPICS.md) | Concepts to study and the reading list |
| [`LEARNINGS.md`](docs/LEARNINGS.md) | Learning goals, predictions-vs-reality log, interview prep |

## Honest scope

This is a **learning-scale** system. It does not compete with production inference engines, and the model is not a general-purpose assistant — TinyStories models write simple children's-story English and little else. The value of the project is the *engineering method*: implement, prove correct, measure, explain the gap between theory and result.

## Acknowledgements and references

- TinyStories dataset and paper — Eldan & Li, 2023
- nanoGPT and llama2.c — Andrej Karpathy (design reference for training loop and tiny Llama-style models)
- Speculative decoding — Leviathan, Kalman & Matias (2022); Chen et al. (2023)
- Full list in [`docs/TOPICS.md`](docs/TOPICS.md)

## License

TBD (MIT suggested). Check the TinyStories dataset card for the dataset's license terms before redistributing any data or derived shards.
