# ---------------------------------------------------------------------------
# Shared YAML / Parquet IO helpers (kept dependency-light: no torch/torch_geometric)
# ---------------------------------------------------------------------------
import os
import yaml
import pandas as pd


def load_yaml_config(config_path: str) -> dict:
    """加载 YAML 配置文件，返回完整字典。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_parquet_append(records: list, output_path: str, columns: list = None):
    """将新记录追加到已有 parquet 文件；文件不存在则新建。

    参数：
        records: 待写入的记录列表（每条为 dict）
        output_path: 目标 parquet 文件路径
        columns: 若提供，仅保留这些列（按给定顺序），用于统一输出 schema
    """
    if not records:
        return
    new_df = pd.DataFrame(records)
    if columns:
        new_df = new_df[[c for c in columns if c in new_df.columns]]
    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        new_df = pd.concat([existing_df, new_df], ignore_index=True)
    new_df.to_parquet(output_path, index=False)
