"""
CoT 数据集构建主程序
=================
功能：
  - 加载本地 vLLM 模型
  - 读取各数据集下 processed_data 下的 with_id.parquet/test.parquet 文件
  - 构造输入 prompts，利用真实标签指导模型生成正确的 CoT 说明
  - 将生成的 CoT 作为新列追加后保存为 parquet 文件
  - 支持断点续跑机制

用法：
  python generation_main.py
  python generation_main.py --config ../configs/cot_gen.yaml
"""

import os
import sys
import argparse
from typing import Union

import yaml
import pandas as pd
from tqdm import tqdm

# 将父目录加入 sys.path，保证引用正常
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path
from utils.prompt_utils import CoTPromptManager
from utils.model_engine import build_engine, VLLMEngine, APIEngine


COT_COL       = "gt_trial"
OUTPUT_SUBDIR = "data_with_cot"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['cot_gen']


# Removed local resolve_path to use utils.path_utils.resolve_path


def save_results(records: list, output_path: str):
    """将缓冲区中的记录追加到输出 parquet 文件。"""
    new_df = pd.DataFrame(records)
    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(output_path, index=False)
    else:
        new_df.to_parquet(output_path, index=False)


def clean_cot_output(raw: str) -> str:
    """
    对大模型输出进行后处理清洗，仅保留最终回答部分：
    过滤掉被 <think>...</think> 或 <|channel>thought ... <channel|> 包裹的思考过程。
    """
    if not raw:
        return ""
    
    import re
    text = raw.strip()
    # 1. 过滤 Qwen / DeepSeek 等推理模型的 <think>...</think> 块
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # 2. 过滤 Gemma 4 等推理模型的 <|channel>thought ... <channel|> 块
    text = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL).strip()
    # 3. 兜底清除可能残留的 <|channel> 或 <channel|> 格式标签
    text = re.sub(r'<\|channel>\w+|<\w+\|>', '', text).strip()
    return text


# ---------------------------------------------------------------------------
# 单数据集处理
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    engine: Union[VLLMEngine, APIEngine],
    engine_type: str,
    pm: CoTPromptManager,
    save_every: int = 100,
):
    """对单个数据集进行 CoT 生成并保存结果。"""
    print(f"\n{'='*60}")
    print(f"[Dataset] {dataset_name}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    df = pd.read_parquet(input_path)

    # 断点续跑：跳过已处理的行
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            if 'id' in existing_df.columns:
                processed_ids = set(existing_df['id'])
                print(f"  [Resume] 已处理 {len(processed_ids)} 条，将跳过")
        except Exception as e:
            print(f"  [Warning] 读取已有输出失败，从头开始: {e}")

    df_todo = df[~df['id'].isin(processed_ids)].copy()

    if len(df_todo) == 0:
        print("  [Done] 所有数据已处理完毕。")
        return

    # 构建所有待处理数据的输入
    rows = df_todo.to_dict('records')
    
    # API 模式下分批处理并实时落盘
    if engine_type == 'api':
        from tqdm import tqdm
        total_saved = 0
        with tqdm(total=len(rows), desc=f"  {dataset_name}", unit="条") as pbar:
            for i in range(0, len(rows), save_every):
                chunk_rows = rows[i : i + save_every]
                chunk_inputs = [
                    pm.get_messages(
                        doc=row['doc'],
                        claim=row['claim'],
                        label=row['label'],
                    )
                    for row in chunk_rows
                ]
                raw_outputs = engine.batch_infer(chunk_inputs, pbar=pbar)

                chunk_records = []
                for row, raw in zip(chunk_rows, raw_outputs):
                    rec = dict(row)
                    rec[COT_COL] = clean_cot_output(raw)
                    chunk_records.append(rec)

                save_results(chunk_records, output_path)
                total_saved += len(chunk_records)
                pbar.set_postfix({"已保存": total_saved})
        print(f"  [Done] 处理完成。共生成 {total_saved} 条 CoT。")
        print(f"  [Done] 结果保存至: {output_path}")

    else:
        all_inputs = [
            pm.get_messages(
                doc=row['doc'],
                claim=row['claim'],
                label=row['label'],
            )
            for row in rows
        ]

        # 一次性喂给推理引擎
        print(f"  [Engine] 正在为 {len(all_inputs)} 条数据生成 CoT ...")
        raw_outputs = engine.batch_infer(all_inputs)

        # 组装记录（带有后处理清洗）
        records_to_save = []
        for row, raw in zip(rows, raw_outputs):
            rec = dict(row)
            rec[COT_COL] = clean_cot_output(raw)
            records_to_save.append(rec)

        # 保存结果
        save_results(records_to_save, output_path)
        total_saved = len(records_to_save)

        print(f"  [Done] 处理完成。共生成 {total_saved} 条 CoT。")
        print(f"  [Done] 结果保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoT Dataset Generator using vLLM")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="配置文件路径，默认使用 configs/cot_gen.yaml"
    )
    args = parser.parse_args()

    # ---- 路径解析 ----
    if args.config:
        config_path = resolve_path(args.config)
    else:
        config_path = resolve_path("configs/cot_gen.yaml")

    if not os.path.exists(config_path):
        print(f"Error: 配置文件不存在 -> {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # ---- 数据集列表 ----
    datasets: list = config['data']['datasets']
    data_root = resolve_path(config['data']['data_root'])
    output_root = resolve_path(config['data']['output_root'])

    # ---- 提示词管理器 ----
    prompts_dir = resolve_path("prompts/cot_gen")
    pm = CoTPromptManager(prompts_dir)

    # ---- 加载推理引擎 ----
    engine_type = config.get('inference', {}).get('engine_type', 'vllm')
    print(f"[Init] 加载推理引擎 (Type: {engine_type}) ...")
    engine = build_engine(config)
    print(f"[Init] 引擎加载完毕。")

    # ---- 推理参数 ----
    test_limit = config['inference'].get('test_limit', 0)
    save_every = config['inference'].get('api', {}).get('save_every_n_items') or config['inference'].get('save_every_n_items', 100)
    output_filename = config['data'].get('output_filename', 'cot_results.parquet')

    # ---- 遍历数据集 ----
    for dataset_name in datasets:
        # minicheck 特殊处理，读取 train.parquet
        if dataset_name.lower() == "minicheck":
            input_file = "train.parquet"
        else:
            input_file = "with_id.parquet"

        input_path  = os.path.normpath(os.path.join(data_root, dataset_name, "processed_data", input_file))
        output_path = os.path.normpath(os.path.join(output_root, dataset_name, OUTPUT_SUBDIR, output_filename))

        if not os.path.exists(input_path):
            print(f"\n[Skip] {dataset_name}: 输入文件不存在 -> {input_path}")
            continue

        # 若设置了测试限制，截取前 N 条
        if test_limit and test_limit > 0:
            df_full = pd.read_parquet(input_path)
            df_full = df_full.head(test_limit).copy()
            tmp_path = input_path.replace(".parquet", "_cot_test_tmp.parquet")
            df_full.to_parquet(tmp_path, index=False)
            actual_input = tmp_path
            print(f"[Data] 测试模式：{dataset_name} 仅处理前 {test_limit} 条")
        else:
            actual_input = input_path

        try:
            process_dataset(
                dataset_name=dataset_name,
                input_path=actual_input,
                output_path=output_path,
                engine=engine,
                engine_type=engine_type,
                pm=pm,
                save_every=save_every,
            )
        finally:
            # 清理临时测试文件
            if test_limit and test_limit > 0 and os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("\n[All Done] 所有数据集处理完毕。")


if __name__ == '__main__':
    main()
