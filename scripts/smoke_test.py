"""End-to-end smoke test on a synthetic toy corpus - works OFFLINE in a few minutes on CPU (or GPU/iGPU).

    python scripts/smoke_test.py                # auto device
    python scripts/smoke_test.py --device xpu   # check your Intel Arc iGPU end to end

It exercises every stage: tokenizer -> data -> train (target + draft) -> eval -> naive/cached/speculative
generation (with equality checks) -> benchmarks E1/E2/E3 -> analysis + plots.
If this passes, the real pipeline will run; the only remaining step is the long training run.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(*args: str) -> None:
    print("\n$ " + " ".join(args), flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
    if r.returncode != 0:
        sys.exit(f"SMOKE TEST FAILED at: {' '.join(args)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-going-on-export-failure", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    dev = ["--device", args.device]

    run("scripts/make_toy_corpus.py", "--out", "data/toy", "--n-train", "4000", "--n-valid", "300")
    run("-m", "tinygpt.tokenizer.train", "--input", "data/toy/train.txt", "--vocab-size", "512", "--out", "data/toy/tokenizer.json")
    run("-m", "tinygpt.data.prepare", "--tok", "data/toy/tokenizer.json", "--train", "data/toy/train.txt",
        "--valid", "data/toy/valid.txt", "--out", "data/toy/shards", "--workers", "1")
    run("-m", "tinygpt.train", "--config", "configs/tiny_smoke.yaml", *dev)
    run("-m", "tinygpt.train", "--config", "configs/tiny_draft_smoke.yaml", *dev)
    run("-m", "tinygpt.eval.perplexity", "--ckpt", "runs/tiny_smoke/model.pt", "--data-dir", "data/toy/shards", *dev,
        "--out", "results/smoke")
    run("-m", "tinygpt.eval.samples", "--ckpt", "runs/tiny_smoke/model.pt", "--tokenizer", "data/toy/tokenizer.json",
        "--data-dir", "data/toy/shards", "--out", "results/smoke/samples.md", "--n-val-prompts", "3",
        "--max-new-tokens", "60", *dev)
    for mode in ("naive", "kvcache"):
        run("-m", "tinygpt.infer.cli", "--ckpt", "runs/tiny_smoke/model.pt", "--tokenizer", "data/toy/tokenizer.json",
            "--mode", mode, "--max-new-tokens", "60", "--temperature", "0", *dev)
    run("-m", "tinygpt.infer.cli", "--ckpt", "runs/tiny_smoke/model.pt", "--draft", "runs/tiny_draft_smoke/model.pt",
        "--tokenizer", "data/toy/tokenizer.json", "--mode", "speculative", "--k", "4", "--max-new-tokens", "60",
        "--temperature", "0", *dev)
    common = ["--target", "runs/tiny_smoke/model.pt", "--data-dir", "data/toy/shards", "--quick", "--prompt-len", "8",
              "--out", "results/smoke", *dev]
    run("-m", "bench.run", "E1", *common, "--lengths", "16,32,64")
    run("-m", "bench.run", "E2", *common, "--new-tokens", "32")
    run("-m", "bench.run", "E3", *common, "--drafts", "tiny_draft=runs/tiny_draft_smoke/model.pt", "--new-tokens", "48")
    run("-m", "bench.analyze", "--results", "results/smoke")
    run("-m", "bench.plot", "--results", "results/smoke")
    try:
        import transformers, tokenizers  # noqa: F401,E401
        run("-m", "export.to_hf_llama", "--ckpt", "runs/tiny_smoke/model.pt", "--tok", "data/toy/tokenizer.json",
            "--out", "runs/tiny_smoke/hf")
    except ImportError:
        print("\n(skipping HF export: pip install transformers tokenizers safetensors to test it)")
    print(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s. Results in results/smoke/. Now run the real pipeline (docs/RUNBOOK.md).")


if __name__ == "__main__":
    main()
