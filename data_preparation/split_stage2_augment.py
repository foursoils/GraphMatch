"""
第二阶段训练数据构建脚本
========================
背景：
  minicheck 训练出的 best_f1.pt 在 AggreFact-CNN / Reveal / ExpertQA 上 BAcc 仅 ~0.55-0.59，
  明显低于其他数据集。为提升这几个数据集的表现，从各自数据集中抽出一部分样本按 label 分层切分为
  train / val / held-out 三份：
    - train + val（各数据集分别按 configs/stage2_augment.yaml 中比例抽取后）合并成跨数据集的
      第二阶段训练/验证集，供 graph_match_nli/train.py 在 minicheck 训练完成的 checkpoint 基础上
      继续微调（continual fine-tuning）。
    - held-out 部分回写到原数据集目录（<source_file>_holdout.parquet），
      graph_match_nli/evaluate.py 推理时会自动优先使用该文件，
      确保最终评估样本与训练/验证样本完全不重叠，避免数据泄露导致指标虚高。

配置文件: configs/stage2_augment.yaml（默认）
用法: python data_preparation/split_stage2_augment.py [配置文件名，位于 configs/ 下]
      例如提高抽样比例的第二版: python data_preparation/split_stage2_augment.py stage2_augment_v2.yaml
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_yaml_config


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'stage2_augment.yaml'
    config_path = os.path.join(base_dir, 'configs', config_name)
    config = load_yaml_config(config_path)['stage2_augment']

    cfg_dir = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    data_root    = resolve(config['data_root'])
    source_file  = config['source_file']
    datasets     = config['datasets']
    train_ratio  = config['train_ratio']
    val_ratio    = config['val_ratio']
    holdout_ratio = 1.0 - train_ratio - val_ratio
    seed         = config.get('seed', 42)
    # holdout_tag：区分不同抽样比例版本的 held-out 文件，避免互相覆盖（如 v1 用默认无 tag，
    # v2/v3 等提高抽样比例的版本各自命名 gemma_26b_tk_holdout_<tag>.parquet）；
    # evaluate.py 读取同一个 config 里的 holdout_tag 来找到对应版本的 held-out 文件
    holdout_tag  = config.get('holdout_tag', '')

    stem = os.path.splitext(source_file)[0]
    holdout_name = f"{stem}_holdout_{holdout_tag}.parquet" if holdout_tag else f"{stem}_holdout.parquet"

    print(f"抽样比例: train={train_ratio:.0%} / val={val_ratio:.0%} / held-out={holdout_ratio:.0%}\n")

    all_train, all_val = [], []
    for ds in datasets:
        src_path = os.path.join(data_root, ds, 'data_with_graph', source_file)
        if not os.path.exists(src_path):
            print(f"[Skip] {ds}: 源文件不存在 -> {src_path}")
            continue

        df = pd.read_parquet(src_path)
        print(f"=== {ds} ===")
        print(f"  原始样本数: {len(df)} | label分布: {df['label'].value_counts().to_dict()}")

        # 第一刀：切出 held-out（70%），剩余 30% 用于 train+val
        df_rest, df_holdout = train_test_split(
            df, test_size=holdout_ratio, random_state=seed, stratify=df['label']
        )
        # 第二刀：剩余部分内部按 train:val 比例再切
        val_relative = val_ratio / (train_ratio + val_ratio)
        df_train, df_val = train_test_split(
            df_rest, test_size=val_relative, random_state=seed, stratify=df_rest['label']
        )

        holdout_path = os.path.join(data_root, ds, 'data_with_graph', holdout_name)
        df_holdout.to_parquet(holdout_path, index=False)

        print(f"  train={len(df_train)} (label={df_train['label'].value_counts().to_dict()}) | "
              f"val={len(df_val)} (label={df_val['label'].value_counts().to_dict()}) | "
              f"held-out={len(df_holdout)} (label={df_holdout['label'].value_counts().to_dict()})")
        print(f"  held-out 已写出 -> {holdout_path}\n")

        df_train = df_train.copy()
        df_val = df_val.copy()
        df_train['source_dataset'] = ds
        df_val['source_dataset'] = ds
        all_train.append(df_train)
        all_val.append(df_val)

    merged_train = pd.concat(all_train, ignore_index=True)
    merged_val = pd.concat(all_val, ignore_index=True)

    train_out = resolve(config['merged_train_output'])
    val_out   = resolve(config['merged_val_output'])
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    merged_train.to_parquet(train_out, index=False)
    merged_val.to_parquet(val_out, index=False)

    print(f"合并后 Stage2 训练集: {len(merged_train)} 条 -> {train_out}")
    print(f"  各数据集来源: {merged_train['source_dataset'].value_counts().to_dict()}")
    print(f"合并后 Stage2 验证集: {len(merged_val)} 条 -> {val_out}")
    print(f"  各数据集来源: {merged_val['source_dataset'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
