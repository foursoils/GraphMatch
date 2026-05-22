"""
评估脚本：计算幻觉检测模型的平衡准确率（BAcc）
=============================================
BAcc = 1/2 * (TPR + TNR)
     = 1/2 * (TP/(TP+FN) + TN/(TN+FP))

注意：pred_label=2 表示解析失败，视为预测错误（不跳过，保留在评估中）。

用法：
  python evaluate.py
  python evaluate.py --config ../configs/evaluation.yaml
"""

import os
import sys
import argparse

import yaml
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['evaluation']


# Removed local resolve_path to use utils.path_utils.resolve_path


def compute_bacc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算平衡准确率及其分解项。

    参数：
        y_true: 真实标签数组（只含 0 和 1）
        y_pred: 预测标签数组（可含 0、1、2，2 视为预测错误）

    返回：包含 TP/TN/FP/FN/TPR/TNR/BAcc 的字典
    """
    # pred_label=2（解析失败）既不是 0 也不是 1，必然判断错误
    # 对于 label=1 的样本，pred=2 → FN；对于 label=0 的样本，pred=2 → FP
    y_pred_mapped = np.where(y_pred == 2, 1 - y_true, y_pred)  # 将 2 映射为与真值相反的预测

    TP = int(np.sum((y_true == 1) & (y_pred_mapped == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred_mapped == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred_mapped == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred_mapped == 0)))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # Recall for positive class
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # Recall for negative class
    BAcc = 0.5 * (TPR + TNR)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "TPR (Recall 1)": TPR,
        "TNR (Recall 0)": TNR,
        "BAcc": BAcc,
    }


def print_confusion_matrix(TP: int, TN: int, FP: int, FN: int):
    total = TP + TN + FP + FN
    print("  混淆矩阵（行=真实, 列=预测）:")
    print(f"               Pred=0    Pred=1")
    print(f"  True=0 (neg)   {TN:5d}     {FP:5d}    | total={TN+FP}")
    print(f"  True=1 (pos)   {FN:5d}     {TP:5d}    | total={FN+TP}")
    print(f"  total          {TN+FN:5d}     {FP+TP:5d}    | N={total}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Evaluation")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="配置文件路径，默认使用 ../configs/evaluation.yaml"
    )
    args = parser.parse_args()

    config_path = args.config or resolve_path("configs/evaluation.yaml")
    config = load_config(config_path)

    label_col = config['columns'].get('label_col', 'label')
    pred_col  = config['columns'].get('pred_col', 'pred_label')
    print_cm  = config['output'].get('print_confusion_matrix', True)
    print_pcr = config['output'].get('print_per_class_recall', True)

    result_files = config['result_files']
    summary_rows = []

    for entry in result_files:
        name = entry['name']
        path = resolve_path(entry['path'])

        print(f"\n{'='*60}")
        print(f"  模型/文件: {name}")
        print(f"  路径:      {path}")
        print(f"{'='*60}")

        if not os.path.exists(path):
            print(f"  [Error] 文件不存在，跳过。")
            continue

        df = pd.read_parquet(path)
        total = len(df)
        parse_fail = int((df[pred_col] == 2).sum())

        print(f"  总样本数:   {total}")
        print(f"  解析失败数: {parse_fail} ({parse_fail/total*100:.1f}%)")

        y_true = df[label_col].values.astype(int)
        y_pred = df[pred_col].values.astype(int)

        metrics = compute_bacc(y_true, y_pred)

        if print_cm:
            print()
            print_confusion_matrix(metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN'])

        print()
        if print_pcr:
            print(f"  TPR (Recall pos=1): {metrics['TPR (Recall 1)']:.4f}")
            print(f"  TNR (Recall neg=0): {metrics['TNR (Recall 0)']:.4f}")

        print(f"\n  ★ BAcc = 1/2 × (TPR + TNR) = {metrics['BAcc']:.4f}")

        summary_rows.append({
            "Name":        name,
            "Total":       total,
            "ParseFail":   parse_fail,
            "TPR":         round(metrics['TPR (Recall 1)'], 4),
            "TNR":         round(metrics['TNR (Recall 0)'], 4),
            "BAcc":        round(metrics['BAcc'], 4),
        })

    # 汇总表格
    if len(summary_rows) > 1:
        print(f"\n{'='*60}")
        print("  汇总")
        print(f"{'='*60}")
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
