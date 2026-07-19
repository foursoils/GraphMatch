# data_preparation

数据准备阶段的脚本集合，按用途大致分三类：**minicheck 基础数据处理**、**Stage2 continual fine-tuning 数据构建**、**辅助/消融工具**。下面按 pipeline 顺序说明每个脚本。

## 1. minicheck 基础数据处理

| 脚本 | 作用 | 配置 | 输出 |
|---|---|---|---|
| `add_id_to_mc.py` | 给原始 minicheck 各分片加上唯一 `id` 列并合并 | `configs/data_prep.yaml`（`minicheck.merge`） | 合并后的 minicheck parquet |
| `split_dataset.py` | 按 label 分层切分为 train/val/test | `configs/data_prep.yaml`（`minicheck.split`） | `train.parquet` / `val.parquet` / `test.parquet` |
| `precompute_embeddings.py` | 用 Embedding 模型（如 Qwen3-Embedding-0.6B）批量预计算图节点/边文本的向量，按样本 id 打包缓存 | `configs/data_prep.yaml` | `data/embeddings/<model_name>/...` |

这一批是训练 Stage1（`configs/graph_match_nli.yaml`）之前的数据预处理，一般只需跑一次。

## 2. Stage2 continual fine-tuning 数据构建（当前最终方案在用）

背景：Stage1（仅在 minicheck 上训练）在 AggreFact-CNN / Reveal / ExpertQA 三个数据集上 BAcc 明显偏低（~0.55-0.59）。这一批脚本用于从这三个数据集里抽样构建 Stage2 continual fine-tuning 的训练/验证数据。

| 脚本 | 作用 | 配置 | 输出 |
|---|---|---|---|
| `split_stage2_augment.py` | 从低分数据集按 label 分层抽样，切出 train/val/held-out 三份；train+val 跨数据集合并成 Stage2 训练集，held-out 回写到源数据集目录，供 `evaluate.py` 评估时自动排除训练样本（防止数据泄露） | `configs/stage2_augment_v2.yaml`（当前用的是 v2：40%/15% 抽样比例）| `data/stage2_augment_v2/train.parquet`、`val.parquet`；各数据集目录下的 `gemma_26b_tk_holdout_v2.parquet` |
| `balance_stage2_train.py` | Stage2 训练集里"支持:幻觉"类别比例失衡（约4.4:1），对幻觉类过采样到 1:1，避免模型只学会"偏向多数类" | 命令行参数指定 `augment_dir` | `data/stage2_augment_v2/train_balanced.parquet` |
| `tune_threshold.py` | 模型训练完之后，在独立验证集（不接触 held-out 测试集）上给每个数据集分别校准"支持概率 > 多少判为支持"的判决阈值，用于在不重新训练、不改模型权重的前提下提升 BAcc | 命令行参数 `--config` / `--ckpt` | 打印校准结果 + 无偏地在 held-out 测试集上验证效果（不落盘，需要手动把选出的阈值填进 `configs/graph_match_nli_stage2_v2.yaml` 的 `data.thresholds`） |

**运行顺序**（复现当前最终方案 `best_f1_stage2_v2.pt` 的数据部分）：

```bash
python data_preparation/split_stage2_augment.py stage2_augment_v2.yaml
python data_preparation/balance_stage2_train.py stage2_augment_v2
# 之后用 configs/graph_match_nli_stage2_v2.yaml 训练，再跑 tune_threshold.py 校准阈值
```

## 3. 已废弃/仅供参考

| 脚本 | 说明 |
|---|---|
| `build_joint_train.py` | 联合训练（把 minicheck 和低分数据集抽样数据合并，从头训练）的探索性方案，用于对比是否能规避 continual fine-tuning 的灾难性遗忘。**实测效果不如 Stage2 continual fine-tuning + 阈值校准，未采用**，脚本保留仅供参考，对应配置和产出已清理。|

## 当前最终方案小结

模型：`models/graph_match/roberta_large_mnli/best_f1_stage2_v2.pt`
配置：`configs/graph_match_nli_stage2_v2.yaml`（Stage2 训练配置 + 阈值校准参数）
数据血缘：`configs/stage2_augment_v2.yaml` → `split_stage2_augment.py` → `balance_stage2_train.py` → 训练 → `tune_threshold.py` 校准
