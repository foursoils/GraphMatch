"""
kg 消融实验主程序
=================
功能：
  - 加载本地 vLLM 模型
  - 读取各数据集下 data_with_graph/gemma_26b_tk.parquet 文件
  - 将 doc / claim 及其三元组图（graph_doc / graph_claim）以文本形式（Hard Prompt）注入 vLLM
  - 将预测结果作为新列追加后保存为 parquet 文件

用法：
  python ablation_main.py
  python ablation_main.py --config ../../configs/ablation.yaml

配置节点：ablation.yaml -> ablation.kg
"""

import os
import sys
import argparse

import yaml
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.path_utils import resolve_path
from utils.prompt_utils import AblationPromptManager
from utils.model_engine import VLLMEngine, parse_binary_label


PRED_COL   = "pred_label"    # 预测结果列名
INPUT_FILE = "gemma_26b_tk.parquet"   # 各数据集下固定读取的文件名
GRAPH_SUBDIR = "data_with_graph"      # 子目录名


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['ablation']['kg']


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


# ---------------------------------------------------------------------------
# 单数据集处理
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    engine: VLLMEngine,
    pm: AblationPromptManager,
):
    """对单个数据集进行幻觉检测推理并保存结果。"""
    print(f"\n{'='*60}")
    print(f"[Dataset] {dataset_name}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    df = pd.read_parquet(input_path)

    # 断点续跑：跳过已处理的行
    processed_ids: set = set()
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

    # 构建所有待处理数据的输入（含图信息）
    rows = df_todo.to_dict('records')
    
    all_inputs = [
        pm.get_messages(
            doc=row['doc'],
            claim=row['claim'],
            graph_doc=row.get('graph_doc', ''),
            graph_claim=row.get('graph_claim', ''),
        )
        for row in rows
    ]

    # 一次性喂给 vLLM，由 vLLM 内部进行连续批处理并显示逐条更新的进度条
    raw_outputs = engine.batch_infer(all_inputs)

    # 解析结果并组装记录
    parse_fail_count = 0
    records_to_save = []
    
    for row, raw in zip(rows, raw_outputs):
        label = parse_binary_label(raw)
        if label == 2:
            parse_fail_count += 1
        rec = dict(row)
        rec[PRED_COL] = label
        records_to_save.append(rec)

    # 一次性保存结果
    save_results(records_to_save, output_path)
    total_saved = len(records_to_save)

    print(f"  [Done] 处理完成。共 {total_saved} 条，解析失败 {parse_fail_count} 条。")
    print(f"  [Done] 结果保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation: Hallucination Detection with Graph-augmented Prompt")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="配置文件路径，默认使用 ../configs/ablation.yaml"
    )
    args = parser.parse_args()

    if args.config:
        config_path = resolve_path(args.config)
    else:
        config_path = resolve_path("configs/ablation.yaml")

    config = load_config(config_path)

    # ---- 数据集列表 ----
    datasets: list = config['data']['datasets']
    data_root = resolve_path(config['data']['data_root'])
    output_root = resolve_path(config['data']['output_root'])

    # ---- 提示词管理器 ----
    prompts_dir = resolve_path("prompts/ablation")
    pm = AblationPromptManager(prompts_dir)

    # ---- 加载推理引擎（仅 vLLM）----
    print(f"[Init] 加载 vLLM 推理引擎 ...")
    engine = VLLMEngine(config)
    print(f"[Init] 引擎加载完毕。")

    # ---- 推理参数 ----
    test_limit = config['inference'].get('test_limit', 0)

    # ---- 遍历数据集 ----
    output_filename = config['data'].get('output_filename', 'ablation_graph_prompt.parquet')

    for dataset_name in datasets:
        input_path  = os.path.join(data_root, dataset_name, GRAPH_SUBDIR, INPUT_FILE)
        output_path = os.path.join(output_root, dataset_name, "ablation_results", output_filename)

        if not os.path.exists(input_path):
            print(f"\n[Skip] {dataset_name}: 输入文件不存在 -> {input_path}")
            continue

        # 若设置了测试限制，截取前 N 条
        if test_limit and test_limit > 0:
            df_full = pd.read_parquet(input_path)
            df_full = df_full.head(test_limit).copy()
            tmp_path = input_path.replace(".parquet", "_test_tmp.parquet")
            df_full.to_parquet(tmp_path, index=False)
            actual_input = tmp_path
            print(f"[Data] 测试模式：{dataset_name} 仅处理前 {test_limit} 条")
        else:
            actual_input = input_path

        process_dataset(
            dataset_name=dataset_name,
            input_path=actual_input,
            output_path=output_path,
            engine=engine,
            pm=pm,
        )

        # 清理临时测试文件
        if test_limit and test_limit > 0 and os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n[All Done] 所有数据集处理完毕。")


if __name__ == '__main__':
    main()
