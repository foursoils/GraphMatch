"""
数据集切分脚本：按配置的比例对输入 Parquet 进行分层切分
配置文件: configs/data_prep.yaml  ← split_dataset 节点
输出: train.parquet, val.parquet, test.parquet
"""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, 'configs', 'data_prep.yaml')
    config = load_config(config_path)['minicheck']['split']

    input_path = os.path.normpath(os.path.join(base_dir, 'configs', config['input_parquet']))
    train_path = os.path.normpath(os.path.join(base_dir, 'configs', config['train_parquet']))
    val_path   = os.path.normpath(os.path.join(base_dir, 'configs', config['val_parquet']))
    test_path  = os.path.normpath(os.path.join(base_dir, 'configs', config['test_parquet']))

    val_ratio  = config.get('val_ratio', 0.1)
    test_ratio = config.get('test_ratio', 0.1)
    seed       = config.get('seed', 42)

    # 校验比例
    total_holdout = val_ratio + test_ratio
    relative_test = test_ratio / total_holdout  # temp 中 test 占比

    df = pd.read_parquet(input_path)
    print(f"总行数: {len(df)}")
    print(f"label 分布:\n{df['label'].value_counts()}\n")

    # 第一刀：切分出 train，剩余为 temp
    df_train, df_temp = train_test_split(
        df, test_size=total_holdout, random_state=seed, stratify=df['label']
    )
    # 第二刀：temp 内按比例分 val / test
    df_val, df_test = train_test_split(
        df_temp, test_size=relative_test, random_state=seed, stratify=df_temp['label']
    )

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"Train : {len(df_train)} 条  | label分布: {df_train['label'].value_counts().to_dict()}")
    print(f"Val   : {len(df_val)}  条  | label分布: {df_val['label'].value_counts().to_dict()}")
    print(f"Test  : {len(df_test)}  条  | label分布: {df_test['label'].value_counts().to_dict()}")
    print(f"\n文件已保存至: {os.path.dirname(train_path)}")


if __name__ == '__main__':
    main()
