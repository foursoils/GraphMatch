# Ablation FT: Qwen3-4B LoRA 文本基线 — Design Spec

**Date:** 2026-07-22  
**Status:** Approved (pending file review)  
**Location:** `ablation/ft/` + `configs/ablation.yaml` → `ablation.ft`

## Goal

在 GraphMatch 消融中增加一条 **纯文本 LoRA 微调基线**：只用 minicheck 的 `(doc, claim) → {0,1}` 监督信号微调 `Qwen3-4B`，不使用任何图结构；再在 minicheck test 与其它数据集上评估，便于与 GraphMatch / 其它消融对照。

## Non-Goals

- 不使用 `data_with_graph`、GMN、embedding 缓存或图注入。
- 不使用 vLLM / TRL SFTTrainer（首版固定 Transformers + PEFT + Accelerate）。
- 不训练 CoT；target 仅为单个字符 `0` 或 `1`。

## Decisions

| Item | Choice |
|------|--------|
| 数据 | `processed_data/{train,val,test}.parquet`；外集 `processed_data/with_id.parquet` |
| Prompt | `prompts/hallu_detect/{system,user}_prompt.txt` |
| 基座 | `models/llms/Qwen3-4B` |
| 适配 | PEFT LoRA（默认 q/k/v/o_proj） |
| 训练/推理引擎 | Transformers + Accelerate |
| 验证指标 | Balanced Accuracy（保存 best） |
| 报告指标 | Acc / BAcc / F1 |

## Layout

```
ablation/ft/
  common.py      # load_config(ablation.ft) / seed / parse_binary_pred
  dataset.py     # HalluTextDataset + collate
  model.py       # Qwen + LoRA；forward(SFT) / inference(generate)
  train.py       # 训练入口
  evaluate.py    # 多数据集推理入口
```

配置节：`configs/ablation.yaml` → `ablation.ft`，字段风格对齐 `configs/graph_match.yaml`：

- `data`: train/val/test 路径、`data_root`、`datasets`、`output_filename`
- `model`: `llm_model_path`、prompt 路径、`max_txt_len`、`max_new_tokens`
- `lora`: r / alpha / dropout / target_modules
- `training`: gpu、bf16、epoch、batch、accum、lr、warmup、patience、output/log dir
- `infer`: batch_size、num_workers、test_limit、gpu_ids

## Data Flow

### Train

1. 读 minicheck train/val parquet。  
2. 用 chat template 拼 system + user（doc/claim）。  
3. 拼接 target `"0"|"1"`；labels 仅在答案 token 上计算（instruction 段 mask 为 -100）。  
4. LoRA 参数 AdamW + cosine warmup；按 val BAcc 存 `best` + `lora_adapter/`。

### Eval

1. 加载基座 `apply_lora=False`，再 `PeftModel.from_pretrained(lora_adapter)`。  
2. 对每个 `datasets` 项：minicheck → test.parquet；其它 → `with_id.parquet`。  
3. `generate(max_new_tokens)` → 解析 `0|1` → 写 `data/<ds>/ablation_results/<output_filename>`，打印 Acc/BAcc/F1。

## Defaults (config)

- LoRA: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, targets `q_proj,k_proj,v_proj,o_proj`
- Train: `num_epochs=1`, `batch_size=1`, `grad_accum_steps=16`, `lr=5e-6`, `bf16`, `patience=3`
- Infer: `batch_size=4`, `max_new_tokens=2`, `test_limit=0`
- Output: `models/ablation_ft/`, log `log/ablation_ft/`

## Risks / Notes

- 长文档需 `max_txt_len` 截断；左 padding 与 GraphMatch 一致，避免 label 错位。  
- 多数据集循环中避免重复 `accelerator.prepare(model)` 导致包装叠加；prepare 一次模型，按数据集换 loader。  
- 模型目录 `models/llms/Qwen3-4B` 需用户事先下载到位。

## Acceptance

- `python ablation/ft/train.py` 能在 minicheck 上完成至少 1 个 epoch 并写出 adapter。  
- `python ablation/ft/evaluate.py` 能在配置的 datasets 上产出 parquet 与指标。  
- 配置全部落在 `ablation.ft`，无需改 `graph_match.yaml`。
