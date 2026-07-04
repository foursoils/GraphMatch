"""
对比实验重试程序
==============
功能：
  - 读取配置文件 hallu_detect.yaml
  - 针对每个数据集下已生成的 contrast_results/*.parquet 文件
  - 筛选出解析失败（pred_label == 2）的样本
  - 重新输入模型进行推理，直到解析成功或达到最大重试次数
  - 将更新后的结果覆写回原 parquet 文件

用法：
  python detect_retry.py
  python detect_retry.py --config ../configs/hallu_detect.yaml
  python detect_retry.py --max_retries 5
"""

import os
import sys
import argparse

import yaml
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path
from utils.prompt_utils import HalluPromptManager
from utils.model_engine import build_engine, parse_binary_label


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['hallu_detect']


# Removed local resolve_path to use utils.path_utils.resolve_path


def main():
    parser = argparse.ArgumentParser(description="Retry contrast experiment predictions that failed parsing.")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--max_retries', type=int, default=3, help="最大重试次数")
    args = parser.parse_args()

    if args.config:
        config_path = resolve_path(args.config)
    else:
        config_path = resolve_path("configs/hallu_detect.yaml")

    config = load_config(config_path)

    # ---- 数据与文件配置 ----
    datasets: list      = config['data']['datasets']
    output_root         = resolve_path(config['data']['output_root'])
    output_filename     = config['data'].get('output_filename', 'contrast_result.parquet')
    engine_type         = config['inference'].get('engine_type', 'vllm').lower()

    # ---- 提示词管理器（NLI 不需要）----
    pm = None
    if engine_type != 'nli':
        prompts_dir = resolve_path("prompts/hallu_detect")
        pm = HalluPromptManager(prompts_dir)

    # 懒加载 engine（只有当真的发现了失败样本时才加载庞大的模型）
    engine = None

    for dataset_name in datasets:
        target_path = os.path.join(output_root, dataset_name, "contrast_results", output_filename)

        if not os.path.exists(target_path):
            print(f"\n[Skip] {dataset_name}: 结果文件不存在 -> {target_path}")
            continue

        df = pd.read_parquet(target_path)
        if 'pred_label' not in df.columns:
            print(f"\n[Skip] {dataset_name}: 不存在 pred_label 列，可能还未运行主程序")
            continue

        # 筛选出失败的样本 (pred_label == 2)
        failed_mask  = df['pred_label'] == 2
        failed_count = failed_mask.sum()

        if failed_count == 0:
            print(f"\n[Skip] {dataset_name}: 没有解析失败的样本。")
            continue

        print(f"\n{'='*60}")
        print(f"[Dataset] {dataset_name} | 发现 {failed_count} 条解析失败的样本待重试。")

        # 若这是第一次发现需要重试的数据，则初始化模型引擎
        if engine is None:
            print(f"[Init] 加载推理引擎: {engine_type} ...")
            engine = build_engine(config)
            print(f"[Init] 引擎加载完毕。")

        # 循环重试
        for attempt in range(1, args.max_retries + 1):
            failed_mask    = df['pred_label'] == 2
            failed_indices = df[failed_mask].index.tolist()
            if not failed_indices:
                break

            print(f"  [Attempt {attempt}/{args.max_retries}] 正在重试 {len(failed_indices)} 条...")

            # 构建输入
            rows = df.loc[failed_indices].to_dict('records')
            if engine_type == 'nli':
                inputs = [(row['doc'], row['claim']) for row in rows]
            else:
                inputs = [
                    pm.get_messages(doc=row['doc'], claim=row['claim'])
                    for row in rows
                ]

            # 推理
            raw_outputs = engine.batch_infer(inputs)

            # 解析并更新原 df
            success_count = 0
            for idx, raw in zip(failed_indices, raw_outputs):
                label = parse_binary_label(raw)
                df.at[idx, 'pred_label'] = label
                if label != 2:
                    success_count += 1

            print(f"    当前轮次成功解析: {success_count} 条 | 仍失败: {len(failed_indices) - success_count} 条")

        # 覆写保存更新后的 dataframe
        df.to_parquet(target_path, index=False)
        final_failed = (df['pred_label'] == 2).sum()
        print(f"  [Done] {dataset_name} 重试结束。最终仍失败数量: {final_failed}。已覆盖保存至: {target_path}")

    print("\n[All Done] 所有数据集重试流程完毕。")


if __name__ == '__main__':
    main()
