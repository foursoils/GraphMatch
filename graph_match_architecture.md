以下是 **GraphMatch（LLM 版）** 的完整架构与技术说明，与 `GraphMatch.tex` Method 章节的双尺度注入设计一致，Backbone 替换为 **Gemma-3-1b-it** 因果语言模型。代码实现位于 `graph_match/` 目录。

## GraphMatch-LLM：双尺度中间层图注入

GraphMatch 将 claim 与 evidence document 表示为关系图，经图匹配网络（GMN）联合编码后，注入 **因果 LLM**（默认 Gemma-3-1b-it）第 $k$ 层内部的两处不同位置：

- **Macro 通路**：在 **Self-Attention 内部**，以 claim–doc 图差异构造 non-uniform attention bias，调制 token 间的因果自注意力分布；
- **Micro 通路**：在 **Self-Attention 输出之后、FFN 之前**，以 Cross-Attention 让 text token 检索 claim/doc 图节点，实现 token–entity 对齐。

与 NLI 分类版（DeBERTa + [CLS] 分类头）不同，LLM 版通过 **SFT + 自回归生成** 输出 `0`/`1` 判断，基座权重冻结，从 `inject_layer` 起应用 **LoRA** 微调。

### 第 $k$ 层前向流程（Gemma Decoder Layer）

```
T^(k-1)  ──►  InputNorm ──►  Self-Attention ──►  H^(k)  ──►  Cross-Attn (Micro) ──►  +residual ──►  PostNorm ──►  FFN ──►  T^(k)
                                  ▲                    ▲
                                  │ Macro: Softmax 前   │ Q = H^(k)
                                  │ 给 AttnLogits 加 bias│ KV = Projector([H_C; H_D])
                                  │                    │
                                  └──── GMN 图级 Δg ───┴── GMN 节点 H_C, H_D
```

Macro 与 Micro 在 **同一层、不同位置** 生效；注入完成后第 $k+1$ 至最终层继续前向，LM Head 生成答案 token。

---

### 0. 输入表示与 GMN 编码

**文本侧**：使用 Gemma chat template 构建 prompt：

```
system: {hallu_detect system prompt}
user:   <doc>...</doc>  <claim>...</claim>
assistant: (训练 target = "0" / "1"，或 CoT + 答案)
```

- `tokenizer.padding_side = 'left'`
- 训练时 `max_txt_len` 默认 **2048**（config 可调）
- 推理时 `max_new_tokens` 默认 **6**（`answer_only` 模式）

**图侧**：LLM 离线从 claim / doc 抽取关系三元组，构建 $G_C$、$G_D$。节点/边文本经 Qwen3-Embedding-0.6B 编码为 **1024 维**初始特征，加载预计算缓存（`train_embed_file` / `val_embed_file`）。

GMN 对两图执行 **3 层**跨图消息传递（默认 hidden **728** 维），输出：

- 节点级：$H_C,\, H_D$ → 合并后经 `GraphProjector` 投影至 LLM 隐维度（Gemma-3-1b-it：**1152**）
- 图级：$g_C,\, g_D = \mathrm{mean\_pool}(H_C),\,\mathrm{mean\_pool}(H_D)$

$$
(H_C,\, H_D,\, g_C,\, g_D) = \mathrm{GMN}(G_C, G_D), \qquad \Delta g = g_C - g_D
$$

---

### 1. Macro 通路：Self-Attention 内的 Non-Uniform Bias

Macro 在 **第 $k$ 层 `self_attn` 模块内部** 生效：在 Softmax 之前向 attention logits 添加图条件偏置（需 `attn_implementation="eager"` 以保证可注入）。

对每个 attention head $h$：

$$
u^{h} = \Delta g \, W_G^{h}, \qquad
b_{ij}^{h} = \frac{(u^{h})^{\top} k_j^{h}}{\sqrt{d_h}}
$$

$$
A_{ij}^{h} = \mathrm{AttnLogits}_{ij}^{h} + \tanh(\alpha_{\mathrm{macro}})\, b_{ij}^{h}
$$

- $k_j^{h}$：RoPE 后的 key 表示（与 Gemma self-attention 内部一致）
- 偏置在 query 维度 $i$ 上共享，随 key 变化 → **non-uniform**
- $\alpha_{\mathrm{macro}}$ 初始化为 **0**

因果 mask 下，仅 $j \le i$ 的有效位置参与 softmax；Macro 使 token 交互受全局图差异条件化。

---

### 2. Micro 通路：Self-Attention 后的 Cross-Attention

Micro 在 **`self_attn` 输出 $H^{(k)}$ 之后、残差相加与 FFN 之前** 生效（`self_attn` forward hook）：

$$
\widetilde{H}^{(k)} = \mathrm{LayerNorm}\!\left(
  H^{(k)} + \tanh(\alpha_{\mathrm{micro}})\,
  \mathrm{CrossAttn}\!\left(H^{(k)},\, \mathrm{Projector}([H_C; H_D])\right)
\right)
$$

- Query = Self-Attention 输出；Key/Value = 投影后的图节点
- 变长节点 padding mask；batch 内样本隔离
- $\alpha_{\mathrm{micro}}$ 初始化为 **0**

---

### 3. 训练与微调策略

| 组件 | 策略 |
|------|------|
| Gemma 基座 | 冻结 |
| LoRA（Layer $k$ 至 $N-1$） | 可训练，`lora_lr = 5e-6` |
| GMN、Projector、Macro bias、Micro Cross-Attn | 全量可训练，`graph_lr = 5e-5` |

**默认超参**（`configs/graph_match.yaml`）：

| 参数 | 值 |
|------|-----|
| LLM | `gemma-3-1b-it`（26 层，hidden=1152，heads=8） |
| 图嵌入 | Qwen3-Embedding-0.6B（1024 维） |
| GMN | 3 层 / hidden 728 |
| 注入层 $k$ | 12 |
| Micro heads | 8 |
| LoRA | r=8, alpha=16, target=q/k/v/o/gate/up/down_proj |
| batch × grad_accum | 1 × 16 |
| 训练 target | `answer_only`（仅监督 `0`/`1`） |
| 辅助 loss | Plan-D cosine（`aux_lambda` 0.2→0.5） |
| 早停指标 | val **BAcc**，`patience=3` |

**主损失**：SFT cross-entropy，仅在 target（答案）token 上计算；prompt 部分 label mask 为 `-100`。

**辅助损失**（可选）：约束 $g_C$ 与 $g_D$ 的余弦相似度方向与 label 一致。

---

### 4. 推理与评估

1. GMN 编码图 → 设置注入上下文
2. Chat prompt → `llm.generate(max_new_tokens=6, do_sample=False)`
3. `parse_binary_pred()` 从生成文本提取 `0`/`1`

LLM 版上下文窗口较长（2048），通常无需 NLI 版的 chunk+max 分块；超长输入在 tokenize 时截断。

批量推理：`python -m graph_match.evaluate`，结果写入各数据集 `our_results/`。指标由 `evaluation/evaluate.py` 计算，主指标 **BAcc**。

**启动命令**：

```bash
# 训练
python graph_match/train.py --config configs/graph_match.yaml

# 推理
python -m graph_match.evaluate --config configs/graph_match.yaml --ckpt models/graph_match/best_model.pt

# 评估
python evaluation/evaluate.py --config configs/evaluation.yaml
```

---

### 5. 与 NLI 版（DeBERTa）的对应关系

| 维度 | NLI 版（论文实验） | LLM 版（本实现） |
|------|-------------------|-----------------|
| Backbone | DeBERTa-v3-large | Gemma-3-1b-it |
| 注入设计 | Macro SA bias + Micro Cross-Attn | **相同** |
| 输出 | [CLS] + 分类头 | 生成 `0`/`1` |
| 训练 loss | NLI 交叉熵 | SFT + 可选 GMN aux |
| 微调 | 冻结前 12 层 + 后层微调 | LoRA + 图模块 |
| 长文档 | chunk + max(prob) | 长上下文截断 / 整段输入 |

双尺度注入机制与 backbone 无关；NLI → LLM 迁移时保留 Macro/Micro 设计，替换输入格式、损失函数与推理协议即可。
