"""Convert the exported HF model to OpenVINO (with KV cache) using optimum-intel, then benchmark (E8).

    pip install "optimum[openvino]"
    python -m export.to_openvino --hf-dir export/hf --out export/ov
    python -m export.bench_external --hf-dir export/hf --ov-dir export/ov --ov-device GPU   # Arc iGPU

Not exercised in the authoring sandbox (no OpenVINO / Intel GPU there): treat the first run on your laptop
as the test, and see docs/RUNBOOK.md for troubleshooting.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="export/hf")
    ap.add_argument("--out", default="export/ov")
    ap.add_argument("--int8", action="store_true", help="also apply OpenVINO 8-bit weight compression")
    args = ap.parse_args()
    from optimum.intel import OVModelForCausalLM
    kwargs = {"export": True, "compile": False}
    if args.int8:
        kwargs["load_in_8bit"] = True
    model = OVModelForCausalLM.from_pretrained(args.hf_dir, **kwargs)
    model.save_pretrained(args.out)
    print(f"[export] OpenVINO model -> {args.out}")


if __name__ == "__main__":
    main()
