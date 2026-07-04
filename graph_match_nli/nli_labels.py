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
    """从 HuggingFace config.id2label 推断 entailment / hallucination 类别索引。"""
    normalized = {int(k): str(v).lower().strip() for k, v in id2label.items()}
    num_labels = len(normalized)

    entailment_id = next(
        (idx for idx, name in normalized.items() if name in {"entailment", "supported", "support"}),
        0,
    )

    if num_labels == 2:
        hallucination_id = next(
            (
                idx for idx, name in normalized.items()
                if idx != entailment_id and ("not" in name or name in {"contradiction", "unsupported"})
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


def dataset_labels_to_nli(labels: torch.Tensor, spec: NLILabelSpec) -> torch.Tensor:
    """
    数据集标签 → NLI 类别索引。
    dataset: 1=支持, 0=幻觉
    """
    labels = labels.view(-1).long()
    if spec.is_binary:
        return 1 - labels

    nli_labels = torch.full_like(labels, fill_value=spec.neutral_id or 1)
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
