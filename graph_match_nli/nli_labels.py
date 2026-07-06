"""NLI 标签映射工具：兼容 2-class 与 3-class 预训练模型。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class NLILabelSpec:
    num_labels: int
    entailment_id: int
    hallucination_id: int
    neutral_id: Optional[int] = None

    @property
    def is_binary(self) -> bool:
        return self.num_labels == 2


def resolve_nli_label_spec(id2label: Dict) -> NLILabelSpec:
    """从 HuggingFace config.id2label 推断 entailment / hallucination 类别索引。

    部分事实核查类 checkpoint（如 MiniCheck 系列）的 id2label 只有裸数字标签
    （如 {0: "0", 1: "1"}），没有语义名称。这类模型在事实核查/grounding 领域
    几乎统一遵循"1=支持(supported/entailed)，0=不支持(unsupported)"的约定
    （与本仓库数据集自身的 label 约定一致），因此把 "1"/"true"/"yes" 也视为
    entailment 关键词、"0"/"false"/"no" 视为 hallucination 关键词，作为语义
    名称匹配失败时的兜底。
    """
    normalized = {int(k): str(v).lower().strip() for k, v in id2label.items()}
    num_labels = len(normalized)

    entailment_id = next(
        (
            idx for idx, name in normalized.items()
            if name in {"entailment", "supported", "support", "true", "yes", "1"}
        ),
        0,
    )

    if num_labels == 2:
        hallucination_id = next(
            (
                idx for idx, name in normalized.items()
                if idx != entailment_id and (
                    "not" in name or name in {"contradiction", "unsupported", "false", "no", "0"}
                )
            ),
            1 - entailment_id,
        )
        return NLILabelSpec(
            num_labels=num_labels,
            entailment_id=entailment_id,
            hallucination_id=hallucination_id,
        )

    contradiction_id = next(
        (idx for idx, name in normalized.items() if name == "contradiction"),
        2,
    )
    neutral_id = next(
        (idx for idx, name in normalized.items() if name == "neutral"),
        None,
    )
    return NLILabelSpec(
        num_labels=num_labels,
        entailment_id=entailment_id,
        hallucination_id=contradiction_id,
        neutral_id=neutral_id,
    )


def default_label_spec(num_labels: int = 2) -> NLILabelSpec:
    """
    未在 NLI/事实核查任务上微调过的纯预训练 backbone（如裸的 microsoft/deberta-v3-large）
    没有 id2label 可供推断语义，分类头也是随机初始化、从零训练。
    这种情况下直接采用与数据集一致的索引约定：1=support(entailment)，0=hallucination，
    避免 resolve_nli_label_spec 对占位符标签（LABEL_0/LABEL_1）做出的语义猜测。
    """
    if num_labels != 2:
        raise ValueError(f"default_label_spec 目前只支持二分类，收到 num_labels={num_labels}")
    return NLILabelSpec(num_labels=2, entailment_id=1, hallucination_id=0)


def dataset_labels_to_nli(labels: torch.Tensor, spec: NLILabelSpec) -> torch.Tensor:
    """
    数据集标签 → NLI 类别索引。
    dataset: 1=支持, 0=幻觉

    必须按 label_spec 的 entailment_id / hallucination_id 映射，不能写死 1-labels：
    MiniCheck 等模型本身就是 1=supported / 0=unsupported，与数据集一致；
    若误用 1-labels 会把监督信号完全反转。
    """
    labels = labels.view(-1).long()
    nli_labels = torch.empty_like(labels)
    nli_labels[labels == 1] = spec.entailment_id
    nli_labels[labels == 0] = spec.hallucination_id
    return nli_labels


def nli_logits_to_support_preds(logits: torch.Tensor, spec: NLILabelSpec) -> torch.Tensor:
    """NLI logits → 数据集预测标签（1=支持, 0=幻觉）。"""
    if spec.is_binary:
        return (logits[:, spec.entailment_id] > logits[:, spec.hallucination_id]).long()

    pred_nli = logits.argmax(dim=-1)
    return (pred_nli == spec.entailment_id).long()


def nli_logits_to_support_probs(logits: torch.Tensor, spec: NLILabelSpec) -> torch.Tensor:
    """返回 entailment（支持）概率。"""
    return torch.softmax(logits, dim=-1)[:, spec.entailment_id]


def build_class_weights(labels_arr, spec: NLILabelSpec, device) -> torch.Tensor:
    """按数据集分布构造 CrossEntropyLoss 类别权重。"""
    pos = int((labels_arr == 1).sum())
    neg = int((labels_arr == 0).sum())
    total = pos + neg

    w_ent = total / (2 * pos) if pos > 0 else 1.0
    w_hal = total / (2 * neg) if neg > 0 else 1.0

    if spec.is_binary:
        weights = [1.0, 1.0]
        weights[spec.entailment_id] = w_ent
        weights[spec.hallucination_id] = w_hal
    else:
        weights = [1.0] * spec.num_labels
        weights[spec.entailment_id] = w_ent
        weights[spec.hallucination_id] = w_hal

    return torch.tensor(weights, dtype=torch.float32, device=device)
