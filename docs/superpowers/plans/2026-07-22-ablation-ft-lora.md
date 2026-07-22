# Ablation FT LoRA Implementation Plan

> **For agentic workers:** Inline execution in this session.

**Goal:** Add pure-text LoRA fine-tuning baseline for Qwen3-4B under `ablation/ft/` with config in `ablation.ft`.

**Architecture:** Mirror `graph_match` train/eval loop without graph modules: PEFT LoRA on Qwen3-4B, SFT on answer token only (`0`/`1`), evaluate with Transformers generate across datasets.

**Tech Stack:** PyTorch, Transformers, PEFT, Accelerate, PyYAML, pandas, scikit-learn

## Global Constraints

- `num_epochs: 1`
- No graph / embedding / GMN
- Data: `processed_data` only
- Prompts: `prompts/hallu_detect/`
- Engine: Transformers + PEFT + Accelerate
- Config key: `ablation.ft` in `configs/ablation.yaml`

---

### Task 1: Config `ablation.ft`

**Files:**
- Modify: `configs/ablation.yaml`

Add `ft:` block with data/model/lora/training/infer mirroring `graph_match.yaml` (epochs=1, Qwen3-4B paths, minicheck processed_data, eval datasets list).

### Task 2: Package helpers + dataset + model

**Files:**
- Create: `ablation/ft/common.py`
- Create: `ablation/ft/dataset.py`
- Create: `ablation/ft/model.py`

### Task 3: Train + evaluate entrypoints

**Files:**
- Create: `ablation/ft/train.py`
- Create: `ablation/ft/evaluate.py`

Prepare model once; swap loaders per dataset during eval.
