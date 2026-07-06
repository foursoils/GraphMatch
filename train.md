以下是针对**基于图匹配网络（GMN）与 NLI 编码器双重注入的事实核查模型（NLI-Graph）**的完整架构与技术说明文段，可直接用于论文的架构设计（Methodology）章节或技术报告中：

## 基于 GMN 双重注入的 NLI 事实核查架构说明

本方案提出了一种 **宏观-微观协同纠偏机制（Macro-Micro Collaborative Correction）** ，将图匹配网络（GMN）提取的结构化拓扑特征注入预训练的 NLI 编码器（DeBERTa-v3），用于判断声明（claim）是否被参考文档（doc）所支持。与将图信息注入自回归大语言模型（LLM）思维链的方案不同，本架构以 **自然语言推理（NLI）分类范式** 为骨干：输入为 `[CLS] doc [SEP] claim [SEP]` 的句对编码，输出为二分类 logits（支持 / 幻觉），更适合高效、可复现的事实核查判别任务。

整体采用双路异构特征融合设计：待检测的断言图（**$G_{\text{claim}}$**）与参考文档子图（**$G_{\text{doc}}$**）经 GMN 跨图传播后，在 DeBERTa 第 **$k$** 层（Self-Attention 之后、FFN 之前）分别从"图级全局拓扑差异"与"节点级细粒度实体对齐"两个维度完成双重注入。

### 0. 输入表示与 GMN 编码

**文本侧**：使用 DeBERTa tokenizer 将 `doc` 与 `claim` 拼接为 NLI 标准句对格式，最大序列长度由配置指定（如 1024）。

**图侧**：由 LLM 预先从 `doc` / `claim` 抽取三元组知识图谱；图中每个节点/边的文本经 SentenceTransformer（如 Qwen3-Embedding-0.6B）编码为 **$D_{\text{emb}}$** 维向量（默认 1024），作为 GMN 的初始节点与边特征。训练时可加载 `data_preparation/precompute_embeddings.py` 预计算的嵌入缓存以加速。

GMN 编码器对 **$G_{\text{claim}}$** 与 **$G_{\text{doc}}$** 执行多层跨图消息传递（默认 3 层），输出：

- 节点级隐状态：**$H_{\text{claim}}, H_{\text{doc}} \in \mathbb{R}^{N \times D_{\text{gmn}}}$**（合并为 **$H_{\text{nodes}}$**）
- 经全局均值池化后的图级向量：**$h_{\text{claim}}, h_{\text{doc}} \in \mathbb{R}^{B \times D_{\text{gmn}}}$**

### 1. 通路一：基于图级差异向量的全局残差注入（Macro-Level）

图级注入旨在从宏观维度让 NLI 编码器感知断言图与文档图之间的整体结构偏离。

计算两图全局向量的差值以捕获拓扑不匹配信号：

$$
\Delta h_G = h_{\text{claim}} - h_{\text{doc}}
$$

通过可学习线性映射 **$W_{\text{graph}} \in \mathbb{R}^{D_{\text{gmn}} \times D_{\text{nli}}}$** 将差值投影至 DeBERTa 隐层维度，并以广播方式融入当前层隐状态 **$H_k \in \mathbb{R}^{B \times L \times D_{\text{nli}}}$**：

$$
H_{\text{macro}} = \text{LayerNorm}\!\left( H_k + \tanh(\alpha_{\text{macro}}) \cdot \text{Unsqueeze}\!\left( \Delta h_G \cdot W_{\text{graph}} \right) \right)
$$

其中 **$\alpha_{\text{macro}}$** 为可学习门控标量，显式初始化为 0。训练初期图级信号不干扰 NLI 编码器原有的语言表征，随优化逐步放大，使模型感知全局事实冲突。

### 2. 通路二：基于节点级矩阵的门控交叉注意力注入（Micro-Level）

节点级注入旨在从微观维度让 NLI 编码器在 token 级别检索结构化实体，实现细粒度事实对齐。

将 GMN 输出的 claim 与 doc 全部节点隐状态横向拼接，构建变长节点矩阵 **$H_{\text{nodes}} \in \mathbb{R}^{N \times D_{\text{gmn}}}$**。以 DeBERTa 第 **$k$** 层 Self-Attention 输出 **$H_k$** 为 Query，经投影后的节点特征为 Key / Value，执行多头缩放点积交叉注意力：

$$
K_g = H_{\text{nodes}} \cdot W_K, \quad V_g = H_{\text{nodes}} \cdot W_V, \quad Q_l = H_k \cdot W_Q
$$

$$
\text{Context} = \text{Softmax}\!\left( \frac{Q_l \cdot K_g^T}{\sqrt{D_{\text{nli}}}} \right) \cdot V_g
$$

对 batch 内每个样本仅使用其自身节点作为 Key/Value（跨样本隔离），并对变长节点数做 padding mask。注入结果经零初始化门控残差融合：

$$
H_{\text{micro}} = \text{LayerNorm}\!\left( H_k + \tanh(\alpha_{\text{micro}}) \cdot \text{Context} \right)
$$

门控标量 **$\alpha_{\text{micro}}$** 同样初始化为 0，保障训练稳定性。微观通路与宏观通路在同一 hook 内顺序执行：先 Cross-Attention 节点注入，再叠加图级差异向量。

注入完成后，DeBERTa 第 **$k+1$** 至最终层继续前向传播；取 **[CLS]** 向量经 Dropout 与线性分类头输出 logits：

$$
\hat{y} = W_{\text{cls}} \cdot \text{Dropout}(H_{\text{final}}[\text{CLS}])
$$

标签约定：**1 = support（支持 / entailment）**，**0 = hallucination（幻觉 / 不支持）**。若加载的 `nli_model_path` 为已在 NLI 任务上微调过的分类 checkpoint，则复用其 `id2label` 语义与分类头权重；若为纯预训练 backbone（如裸 `deberta-v3-large`），分类头随机初始化。

### 3. 训练与微调策略

本架构采用 **分层冻结 + 差异化学习率** 的高效微调方案，而非 LoRA：

| 组件 | 策略 |
|------|------|
| DeBERTa 前 **$N$** 层（如 `freeze_nli_layers=12`） | 完全冻结，保留通用语言理解能力 |
| DeBERTa 后 **$L-N$** 层 | 参与训练，学习率 = `learning_rate × deberta_lr_ratio`（通常更低） |
| GMN、Cross-Attention 注入层、图级投影、分类头 | 端到端联合训练，学习率 = `learning_rate` |

**损失函数**：带类别权重的交叉熵（`CrossEntropyLoss`），可选 label smoothing（如 0.05），在 NLI 标签空间计算以与 `id2label` 映射一致。

**优化器**：AdamW + Linear Warmup → Cosine Decay 学习率调度。

**训练监控**：以验证集 **F1** 为早停指标（`patience` 连续若干 epoch 无提升则停止），保存最优检查点至 `models/graph_match/best_f1.pt`。

**数据**：默认在 minicheck 的 `train / val / test` 划分上训练（`configs/graph_match_nli.yaml`），图数据来自 `data_with_graph/<generator>/` 目录。

**多卡**：配置 `training.tensor_parallel_size > 1` 时自动拉起 DDP 多进程训练；亦兼容 `torchrun` 启动。

### 4. 推理与评估

训练完成后，使用 `graph_match_nli/evaluate.py` 加载检查点，对各 benchmark 批量推理，将 `pred_label` / `pred_prob` 写入各数据集的 `our_results/` 目录。最终指标由 `evaluation/evaluate.py` 统一计算，主指标为 **BAcc**（平衡准确率）。

**启动命令示例**：

```bash
# 训练
python graph_match_nli/train.py --config configs/graph_match_nli.yaml

# 推理
python graph_match_nli/evaluate.py --ckpt models/graph_match/best_f1.pt

# 评估
python evaluation/evaluate.py --config configs/evaluation.yaml
```
