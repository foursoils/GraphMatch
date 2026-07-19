"""
联合训练集构建脚本
==================
背景：
  continual fine-tuning（先在 minicheck 上训练，再在低分数据集抽样数据上继续训练）出现了
  灾难性遗忘：minicheck 及其他未涉及的数据集指标全线下降，抽样数据集本身的 BAcc 也没有实质提升。

  改为联合训练：把 minicheck 的 train/val 与 AggreFact-CNN/Reveal/ExpertQA 抽样出的
  train/val（见 data_preparation/split_stage2_augment.py 生成的 data/stage2_augment/）
  直接合并成一份更大的训练/验证集，从 roberta-large-mnli 原始权重重新训练一个模型，
  而不是在已收敛的 checkpoint 上继续训练，避免遗忘之前学到的知识。

输出:
  data/joint_train/train.parquet
  data/joint_train/val.parquet
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mc_train = pd.read_parquet(os.path.join(BASE_DIR, 'data/minicheck/data_with_graph/gemma_26b_tk/train.parquet'))
    mc_val   = pd.read_parquet(os.path.join(BASE_DIR, 'data/minicheck/data_with_graph/gemma_26b_tk/val.parquet'))
    mc_train['source_dataset'] = 'minicheck'
    mc_val['source_dataset']   = 'minicheck'

    aug_train = pd.read_parquet(os.path.join(BASE_DIR, 'data/stage2_augment/train.parquet'))
    aug_val   = pd.read_parquet(os.path.join(BASE_DIR, 'data/stage2_augment/val.parquet'))

    joint_train = pd.concat([mc_train, aug_train], ignore_index=True)
    joint_val   = pd.concat([mc_val, aug_val], ignore_index=True)

    out_dir = os.path.join(BASE_DIR, 'data', 'joint_train')
    os.makedirs(out_dir, exist_ok=True)
    joint_train.to_parquet(os.path.join(out_dir, 'train.parquet'), index=False)
    joint_val.to_parquet(os.path.join(out_dir, 'val.parquet'), index=False)

    print(f"联合训练集: {len(joint_train)} 条 -> {out_dir}/train.parquet")
    print(f"  来源分布: {joint_train['source_dataset'].value_counts().to_dict()}")
    print(f"  label分布: {joint_train['label'].value_counts().to_dict()}")
    print(f"联合验证集: {len(joint_val)} 条 -> {out_dir}/val.parquet")
    print(f"  来源分布: {joint_val['source_dataset'].value_counts().to_dict()}")
    print(f"  label分布: {joint_val['label'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
