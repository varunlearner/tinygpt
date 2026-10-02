# Runbook - every command, in order

**Goal:** run the commands below top to bottom. Everything except the training runs themselves is already written and tested;
the only "real work" left is waiting for the training to finish.

> **Already verified (in a CPU-only sandbox):** the full unit-test suite (tokenizer, model, KV-cache equivalence, speculative-decoding
> greedy identity + statistical distribution test, exact training resume, HF-Llama export parity, int8), the offline smoke test
> (`scripts/smoke_test.py`), the real 28M config instantiating and taking training steps, and `torch.compile` on the decode step.
>
> **Not verifiable there - run the check steps first and report problems:** CUDA (T4) execution, Intel XPU (Arc iGPU) execution,
> OpenVINO export, llama.cpp/GGUF conversion, the Kaggle notebooks' file-path detection, and the actual download of TinyStories
> (no access to huggingface.co from the authoring sandbox). Section 7 lists likely first-run issues and fixes.

Timing numbers below are **estimates** - your own logs are the truth.

---

## 0. One-time setup

### Laptop (Windows, Core Ultra 7 155H / Arc iGPU) - PowerShell

```powershell
cd tinygpt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/xpu      # Intel GPU build (check pytorch.org "Intel GPU" for the current command)
pip install -r requirements.txt
pip install transformers tokenizers safetensors                        # for export + parity tests
pip install -e .
python scripts/probe_xpu.py xpu                                        # note TFLOPS + GB/s in docs/ENVIRONMENT.md
```

If `python scripts/probe_xpu.py xpu` says XPU is unavailable: update the Intel graphics driver, reinstall the XPU wheel, and
keep going with `cpu` - everything works on CPU, just slower. (Only the iGPU benchmark leg needs XPU.)

### Linux / macOS / WSL

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                       # or the CPU/XPU wheel from pytorch.org
pip install -r requirements.txt && pip install transformers tokenizers safetensors && pip install -e .
```

### Kaggle

1. Upload `tinygpt.zip` as a **Dataset** (Datasets -> New Dataset). Kaggle unpacks it automatically.
2. Import `notebooks/01_prepare_data_cpu.ipynb` and `notebooks/02_train_and_benchmark_gpu.ipynb` (File -> Import Notebook).
3. In each notebook: **Add Data** -> attach the code dataset; enable **Internet** (Settings).

---

## 1. Verify everything works (laptop, ~5-10 min)

```powershell
python -m pytest                              # full suite (~3-5 min on CPU; the two statistical tests are the slow ones)
python scripts/smoke_test.py --device cpu     # full tiny pipeline: tokenizer -> data -> train -> generate -> bench -> plots (~1-3 min)
python scripts/smoke_test.py --device xpu     # same on the Arc iGPU - proves your XPU setup end to end
```

`SMOKE TEST PASSED` means the real pipeline will run. Smoke outputs go to `data/toy`, `runs/tiny_*`, `results/smoke` (safe to delete).

---

## 2. Get the data (Kaggle CPU notebook recommended - it does not use GPU quota)

**Kaggle:** run `01_prepare_data_cpu.ipynb` (Accelerator: *None*), then **Save Version -> Save & Run All (Commit)**.

**Or locally** (needs ~3 GB disk for raw text + ~1.3 GB shards; 4+ cores help):

```powershell
python -m tinygpt.data.download --out data/raw                 # ~2.2 GB. Quick test: add  --train-mb 300
python -m tinygpt.tokenizer.train --input data/raw/TinyStoriesV2-GPT4-train.txt --max-bytes 200000000 --vocab-size 4096 --out data/tok/tokenizer.json
python scripts/tokenizer_parity.py --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt --tok data/tok/tokenizer.json
python -m tinygpt.data.prepare --tok data/tok/tokenizer.json --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt --out data/shards --workers 4
```

- Expect: ~2.7M training stories; a few hundred million tokens (**the exact count is printed and saved to `data/shards/meta.json` - put it in the README**).
- `tokenizer_parity.py` should report compression within ~2% of Hugging Face `tokenizers` (PRD FR-T4). If it is off by more, record it in `DECISIONS.md` (ADR-007).
- Peak RAM stays small (the script streams). On a 16 GB laptop use `--workers 2` if the machine gets sluggish.

---

## 3. TRAIN  <- the only real work

### 3a. Check throughput and (optionally) pick a learning rate

**Kaggle:** notebook 02 with `STAGE = 'check'` (runs tests, GPU smoke test, a 60-step throughput probe of the real config).
Read the `k tok/s` in the log. `epochs: 1` of the data at that speed tells you the training time; compare it with your remaining weekly GPU quota.

Optional LR sweep (~15 min on a T4) - pick a value slightly below the best:

```bash
python scripts/lr_sweep.py --config configs/target_28m.yaml --steps 300 --lrs 3e-4,6e-4,1e-3,2e-3,3e-3
```

If the best LR differs from 1e-3, edit `train.lr` in `configs/target_28m.yaml`.

### 3b. Train the target (28M) - resumable

**Kaggle:** notebook 02, `STAGE = 'target'`, **Save & Run All (Commit)**. It stops cleanly before the session limit; re-run the
notebook (attach the previous run's output as data) and it resumes. Repeat until the log says `done`.

**Local / Colab / any GPU:**

```bash
python -m tinygpt.train --config configs/target_28m.yaml --compile
python -m tinygpt.train --config configs/target_28m.yaml --compile --resume     # after a crash / session end
python -m tinygpt.train --config configs/target_28m.yaml --compile --resume --time-limit-minutes 660   # stop gracefully before a session limit
```

What to watch: loss should fall steadily (a healthy run ends with val loss far below the uniform-guess level `ln(4096) = 8.32`);
`gnorm` staying around ~1 or below; no `NaN`. On NaN/inf the script stops with an error - `--resume` from the last checkpoint and lower `train.lr`.
Outputs: `runs/target_28m/{model.pt, model_best.pt, ckpt_latest.pt, train_log.csv, eval_log.csv, sample.txt, run_summary.json}`.

### 3c. Train the draft models (much cheaper)

```bash
python -m tinygpt.train --config configs/draft_4m.yaml --compile           # 4-layer draft
python -m tinygpt.train --config configs/draft_2l.yaml --compile           # 2-layer draft (ablation)
python -m tinygpt.train --config configs/draft_4m_distilled.yaml --compile # distilled from the target (needs runs/target_28m/model.pt)
```

**Kaggle:** notebook 02, `STAGE = 'drafts'`.

### 3d. Optional extras

```bash
python -m tinygpt.train --config configs/target_28m_gqa.yaml --compile     # GQA ablation (4x smaller KV cache)
python -m tinygpt.train --config configs/target_42m.yaml --compile         # 42M stretch (may need micro_batch_size 16 + grad_accum 8 on a T4)
```

---

## 4. Evaluate

```bash
python -m tinygpt.eval.perplexity --ckpt runs/target_28m/model.pt --data-dir data/shards --dtype fp16
python -m tinygpt.eval.samples --ckpt runs/target_28m/model.pt --tokenizer data/tok/tokenizer.json --data-dir data/shards --out results/samples/target_28m.md
python -m tinygpt.infer.cli --ckpt runs/target_28m/model.pt --prompt "One day, a little dog" --mode kvcache
python -m tinygpt.infer.cli --ckpt runs/target_28m/model.pt --draft runs/draft_4m_distilled/model.pt --mode speculative --k 4 --prompt "One day, a little dog"
```

Read `results/samples/target_28m.md` yourself (fixed prompts, no cherry-picking) and note quality in `docs/LEARNINGS.md`.

---

## 5. Benchmark (this is the headline of the project)

**Write your predictions in `docs/LEARNINGS.md` first** (section 2), then run:

```bash
# on Kaggle T4 (notebook 02, STAGE='bench') or any CUDA machine:
python -m bench.run E1 --target runs/target_28m/model.pt                    # naive vs KV cache vs length
python -m bench.run E2 --target runs/target_28m/model.pt                    # batch scaling
python -m bench.run E3 --target runs/target_28m/model.pt --drafts draft_2l=runs/draft_2l/model.pt,draft_4l=runs/draft_4m/model.pt,draft_4l_distilled=runs/draft_4m_distilled/model.pt
python -m bench.run E5 --target runs/target_28m/model.pt --draft runs/draft_4m/model.pt   # eager vs torch.compile (CUDA graphs)
python -m bench.analyze --results results      # E4 theory-vs-measured + E6 roofline/MBU
python -m bench.plot --results results         # figures + results/SUMMARY.md
```

**On your laptop** (copy `runs/*/model.pt`, `data/shards/meta.json`, `data/shards/val.bin` and `data/tok/tokenizer.json` from Kaggle; `train.bin` is not needed):

```powershell
python -m bench.run E1 --target runs/target_28m/model.pt --device xpu --power-mode "plugged-in/best-performance"
python -m bench.run E1 --target runs/target_28m/model.pt --device cpu  --power-mode "plugged-in/best-performance" --naive-max 128 --runs 5
python -m bench.run E3 --target runs/target_28m/model.pt --device xpu --drafts draft_4l=runs/draft_4m/model.pt,draft_4l_distilled=runs/draft_4m_distilled/model.pt
python -m bench.analyze --results results
python -m bench.plot --results results
```

Benchmark hygiene (details in `docs/BENCHMARKING.md`): laptop plugged in, "Best performance" power mode, other apps closed, no
throttling; the harness warns when a result's IQR exceeds 10% of its median - re-run those. CPU naive generation at 480 tokens is slow:
use `--naive-max 128` and `--runs 5` on CPU.

Then copy the numbers from `results/SUMMARY.md` into the README table (each row cites its experiment ID + JSON file), and write the
"why" paragraphs from `docs/BENCHMARKING.md` section 5.

---

## 6. Export, quantize, compare (optional but strong)

```bash
python -m export.to_hf_llama --ckpt runs/target_28m/model.pt --tok data/tok/tokenizer.json --out export/hf     # logit parity check built in
python -m export.quantize --ckpt runs/target_28m/model.pt --data-dir data/shards                               # E7: int8 quality/speed

# OpenVINO on the Arc iGPU (needs: pip install "optimum[openvino]")
python -m export.to_openvino --hf-dir export/hf --out export/ov
python -m export.bench_external --native runs/target_28m/model.pt --hf-dir export/hf --ov-dir export/ov --ov-device GPU --device cpu   # E8

# llama.cpp (GGUF)
git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py export/hf --outfile export/tinygpt-f16.gguf --outtype f16
# build llama.cpp (see its README; add the Vulkan or SYCL backend for the iGPU), then e.g.:
#   llama-bench -m export/tinygpt-f16.gguf -p 16 -n 128
```

Publish (optional): `huggingface-cli upload <you>/tinygpt-28m export/hf .` and include a model card listing limitations
(children's-story English only; the TinyStories dataset license is CDLA-Sharing-1.0 - check the dataset card).

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `CUDA out of memory` while training | lower `train.micro_batch_size` and raise `train.grad_accum_steps` to keep the product constant (e.g. 16 x 8) |
| Loss becomes NaN / `RuntimeError: loss is NaN/inf` | fp16 overflow or LR too high: `--resume` from the last checkpoint with `--set train.lr=5e-4` (or `--set train.dtype=fp32` to debug) |
| Training very slow | first ~1-2 min with `--compile` is compilation; if tok/s stays far below expectation, drop `--compile` and compare |
| `--compile` fails on a platform | run without `--compile` (it is optional) |
| `torch.xpu.is_available()` is False | update the Intel graphics driver; reinstall the XPU wheel; check `python -c "import torch; print(torch.__version__)"` ends in `+xpu` |
| XPU first run takes minutes | kernel JIT compilation; the harness warms up (first numbers are discarded) |
| `torch.Generator(device='xpu')` errors | handled automatically (falls back to the global RNG) |
| Kaggle notebook: "Code not found" | attach the tinygpt.zip dataset via *Add Data*; it must contain `pyproject.toml` |
| Kaggle notebook: "Attach the output of notebook 01" | commit notebook 01 first, then *Add Data -> Notebook Output Files* in notebook 02 |
| Speculative decoding shows speedup < 1 | expected on very fast models: the draft is not cheap enough (`c` close to 1). Check E5 (`torch.compile` lowers `c`), use the 2-layer draft, increase k - and **report it with the E4 analysis**; that is a legitimate result |
| GGUF conversion error about an unknown BPE pre-tokenizer | in `convert_hf_to_gguf.py`, `get_vocab_base_pre` does not recognize custom tokenizers; make it return `"gpt-2"` for this model (our regex is GPT-2's) |
| OpenVINO export fails | try `pip install -U optimum[openvino]`; run the export on CPU first; the NPU generally cannot run a growing KV cache |
| `tokenizer_parity.py` differs >2% | record the number and reason in `docs/DECISIONS.md`; small tie-breaking differences are expected |

## 8. After the runs - finish the docs

- [ ] README results table filled from `results/SUMMARY.md` (every row cites an experiment ID + JSON)
- [ ] `docs/LEARNINGS.md`: predictions vs actual, bug museum, interview answers in your own words
- [ ] `docs/ENVIRONMENT.md`: probe results table
- [ ] `docs/DECISIONS.md`: any decision that changed (vocab size, LR, etc.)
- [ ] Commit `results/*.json`, `results/plots/`, `results/samples/`; do **not** commit weights/data (see `.gitignore`)
