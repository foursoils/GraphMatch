"""
Stage2 训练集类别重采样脚本
==========================
背景：
  data/stage2_augment/train.parquet（由 split_stage2_augment.py 从 AggreFact-CNN/Reveal/ExpertQA
  抽样合并而来）支持:幻觉 ≈ 955:216（约 4.4:1）。之前两轮 continual fine-tuning 实验显示，
  模型在这种不均衡小样本上主要学到的是"更倾向预测多数类"，Val AUC 几乎不涨（0.567→0.573），
  说明 loss 里的 class_weight 补偿力度不够，需要在数据层面直接重采样。

做法：
  对幻觉类（少数类）做过采样（随机重复采样，允许重复），使其数量与支持类（多数类）持平，
  合并后 shuffle。验证集（val.parquet）保持原始分布不变（早停指标按真实分布评估更有意义）。

输出:
  data/<augment_dir>/train_balanced.parquet（默认 augment_dir=stage2_augment）

用法: python data_preparation/balance_stage2_train.py [augment_dir]
      例如更高抽样比例的第二版: python data_preparation/balance_stage2_train.py stage2_augment_v2
"""
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42


def main():
    augment_dir = sys.argv[1] if len(sys.argv) > 1 else 'stage2_augment'
    train_path = os.path.join(BASE_DIR, 'data', augment_dir, 'train.parquet')
    df = pd.read_parquet(train_path)

    counts = df['label'].value_counts()
    print(f"原始训练集: {len(df)} 条 | label分布: {counts.to_dict()}")

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    n_majority = counts[majority_label]
    n_minority = counts[minority_label]

    df_majority = df[df['label'] == majority_label]
    df_minority = df[df['label'] == minority_label]

    # 对少数类过采样（有放回抽样）到与多数类持平
    df_minority_upsampled = df_minority.sample(
        n=n_majority, replace=True, random_state=SEED
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = os.path.join(BASE_DIR, 'data', augment_dir, 'train_balanced.parquet')
    df_balanced.to_parquet(out_path, index=False)

    print(f"过采样后: {len(df_balanced)} 条 | label分布: {df_balanced['label'].value_counts().to_dict()}")
    print(f"  少数类(label={minority_label}) 原始 {n_minority} 条，过采样重复率 ≈ {n_majority / n_minority:.2f}x")
    print(f"  来源分布: {df_balanced['source_dataset'].value_counts().to_dict()}")
    print(f"已写出 -> {out_path}")


if __name__ == '__main__':
    main()
