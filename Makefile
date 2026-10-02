# Convenience targets for Linux / macOS / WSL / Kaggle. On Windows, run the same commands from docs/RUNBOOK.md.
PY      ?= python
DEVICE  ?= auto
WORKERS ?= 4
TRAIN_TXT = data/raw/TinyStoriesV2-GPT4-train.txt
VALID_TXT = data/raw/TinyStoriesV2-GPT4-valid.txt

.PHONY: help test smoke download tokenizer parity prepare data lr-sweep train-target train-draft train-draft-2l \
        train-distilled eval samples bench-e1 bench-e2 bench-e3 bench-e5 analyze plot export clean

help:
	@echo "make test | smoke | data | train-target | train-draft | train-distilled | eval | bench-e1 ... | analyze | plot | export"

test:            ; $(PY) -m pytest
smoke:           ; $(PY) scripts/smoke_test.py --device $(DEVICE)

download:        ; $(PY) -m tinygpt.data.download --out data/raw
tokenizer:       ; $(PY) -m tinygpt.tokenizer.train --input $(TRAIN_TXT) --max-bytes 200000000 --vocab-size 4096 --out data/tok/tokenizer.json
parity:          ; $(PY) scripts/tokenizer_parity.py --train $(TRAIN_TXT) --valid $(VALID_TXT) --tok data/tok/tokenizer.json
prepare:         ; $(PY) -m tinygpt.data.prepare --tok data/tok/tokenizer.json --train $(TRAIN_TXT) --valid $(VALID_TXT) --out data/shards --workers $(WORKERS)
data: download tokenizer prepare

lr-sweep:        ; $(PY) scripts/lr_sweep.py --config configs/target_28m.yaml --steps 300 --device $(DEVICE)

# ---- THE TRAINING STEP (all resumable: add RESUME=--resume) -------------------------------------
train-target:    ; $(PY) -m tinygpt.train --config configs/target_28m.yaml --device $(DEVICE) --compile $(RESUME)
train-draft:     ; $(PY) -m tinygpt.train --config configs/draft_4m.yaml --device $(DEVICE) --compile $(RESUME)
train-draft-2l:  ; $(PY) -m tinygpt.train --config configs/draft_2l.yaml --device $(DEVICE) --compile $(RESUME)
train-distilled: ; $(PY) -m tinygpt.train --config configs/draft_4m_distilled.yaml --device $(DEVICE) --compile $(RESUME)

eval:
	$(PY) -m tinygpt.eval.perplexity --ckpt runs/target_28m/model.pt --data-dir data/shards --device $(DEVICE) --dtype fp16
	$(PY) -m tinygpt.eval.perplexity --ckpt runs/draft_4m/model.pt --data-dir data/shards --device $(DEVICE) --dtype fp16
samples:
	$(PY) -m tinygpt.eval.samples --ckpt runs/target_28m/model.pt --out results/samples/target_28m.md --device $(DEVICE)

DRAFTS = draft_2l=runs/draft_2l/model.pt,draft_4l=runs/draft_4m/model.pt,draft_4l_distilled=runs/draft_4m_distilled/model.pt
bench-e1:        ; $(PY) -m bench.run E1 --target runs/target_28m/model.pt --device $(DEVICE)
bench-e2:        ; $(PY) -m bench.run E2 --target runs/target_28m/model.pt --device $(DEVICE)
bench-e3:        ; $(PY) -m bench.run E3 --target runs/target_28m/model.pt --drafts $(DRAFTS) --device $(DEVICE)
bench-e5:        ; $(PY) -m bench.run E5 --target runs/target_28m/model.pt --draft runs/draft_4m/model.pt --device $(DEVICE)
analyze:         ; $(PY) -m bench.analyze --results results
plot:            ; $(PY) -m bench.plot --results results

export:
	$(PY) -m export.to_hf_llama --ckpt runs/target_28m/model.pt --tok data/tok/tokenizer.json --out export/hf

clean:           ; rm -rf runs data/toy results/smoke .pytest_cache
