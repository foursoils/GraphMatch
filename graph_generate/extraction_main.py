import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path
from utils.prompt_utils import PromptManager
from utils.model_engine import LocalQwenExtractor
from utils.io_utils import load_yaml_config, save_parquet_append

load_config = load_yaml_config

GRAPH_BUFFER_COLUMNS = ['id', 'claim', 'graph_claim', 'claim_graph_status', 'doc', 'graph_doc', 'doc_graph_status', 'label']


def save_buffer(buffer: list, output_path: str):
    """保存并追加缓冲区数据到 parquet 文件。"""
    save_parquet_append(buffer, output_path, columns=GRAPH_BUFFER_COLUMNS)

def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    extractor: LocalQwenExtractor,
    pm: PromptManager,
    batch_size: int,
    save_every_n_items: int,
):
    """对单个数据集运行知识图谱三元组提取。"""
    print(f"\n{'='*60}")
    print(f"[Dataset] {dataset_name}")
    print(f"  输入路径: {input_path}")
    print(f"  输出路径: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)

    # 支持断点续跑逻辑
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            if 'id' in existing_df.columns:
                processed_ids = set(existing_df['id'])
                print(f"  [Resume] 探测到历史生成文件，已处理 {len(processed_ids)} 条，将跳过。")
        except Exception as e:
            print(f"  [Warning] 读取已有 Parquet 失败: {e}")
                    
    df_to_process = df[~df['id'].isin(processed_ids)].copy()

    if len(df_to_process) == 0:
        print("  [Done] 所有数据的图谱均已提取保存，无需重新提取。")
        return

    new_records_buffer = []
    pbar = tqdm(total=len(df_to_process), desc="  提取图谱结构中", unit="条")
    total_saved = 0

    # 组装所有子任务 (每个 row 包括 doc 和 claim 两个 LLM 提取请求)
    jobs = []
    for idx, row in df_to_process.iterrows():
        jobs.append((idx, 'doc', pm.get_messages(str(row['doc']))))
        jobs.append((idx, 'claim', pm.get_messages(str(row['claim']))))
        
    pending_rows = {}
    
    # 批量提取
    actual_batch_size = batch_size if batch_size is not None else (len(jobs) if jobs else 1)
    for i in range(0, len(jobs), actual_batch_size):
        batch_jobs = jobs[i:i+actual_batch_size]
        batch_msgs = [j[2] for j in batch_jobs]
        
        res_list = extractor.batch_extract(batch_msgs)
        
        for j, res in zip(batch_jobs, res_list):
            idx, target_type, _ = j
            
            if idx not in pending_rows:
                pending_rows[idx] = {}
            pending_rows[idx][target_type] = res
            
            # 当 doc 与 claim 的三元组均提取完毕
            if 'doc' in pending_rows[idx] and 'claim' in pending_rows[idx]:
                row = df_to_process.loc[idx]
                doc_res = pending_rows[idx]['doc']
                claim_res = pending_rows[idx]['claim']
                
                status_doc, res_doc = extractor._parse_graph_tag(doc_res)
                status_claim, res_claim = extractor._parse_graph_tag(claim_res)
                
                rec = row.to_dict()
                rec['doc_graph_status'] = status_doc
                rec['graph_doc'] = json.dumps(res_doc, ensure_ascii=False)
                rec['claim_graph_status'] = status_claim
                rec['graph_claim'] = json.dumps(res_claim, ensure_ascii=False)
                
                new_records_buffer.append(rec)
                pbar.update(1)
                del pending_rows[idx]
                
                if len(new_records_buffer) >= save_every_n_items:
                    save_buffer(new_records_buffer, output_path)
                    total_saved += len(new_records_buffer)
                    pbar.set_postfix({"已落盘": f"{total_saved}/{len(df_to_process)}"})
                    new_records_buffer = []
                    
    # 保存尾部残留数据
    if len(new_records_buffer) > 0:
        save_buffer(new_records_buffer, output_path)
        total_saved += len(new_records_buffer)
        
    pbar.close()
    print(f"  [Done] 数据集 {dataset_name} 图谱提取完成。")


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph triples using local or API models.")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="配置文件路径，默认使用 configs/graph_gen.yaml"
    )
    args = parser.parse_args()

    if args.config:
        config_path = resolve_path(args.config)
    else:
        config_path = resolve_path("configs/graph_gen.yaml")
        
    print(f"[Init] 加载配置文件: {config_path}")
    config = load_config(config_path)['minicheck']
    
    # 提取配置参数
    data_root = resolve_path(config['data']['data_root'])
    output_root = resolve_path(config['data']['output_root'])
    input_subdir = config['data'].get('input_subdir', 'processed_data')
    input_filename = config['data'].get('input_filename', 'with_id.parquet')
    output_subdir = config['data'].get('output_subdir', 'data_with_graph')
    output_filename = config['data'].get('output_filename', 'gemma_26b_tk.parquet')
    datasets = config['data']['datasets']

    prompts_dir = resolve_path("prompts/graph_gen")
    
    # 仅初始化模型和 Prompt 管理器一次
    extractor = LocalQwenExtractor(config)
    pm = PromptManager(prompts_dir)
    
    engine_type = config['inference'].get('engine_type', 'api').lower()
    if engine_type == 'api':
        batch_size = config['inference'].get('api', {}).get('batch_size', config['inference'].get('batch_size', 20))
    else:
        # vLLM 模式下，不需要在 Python 层切小 batch，直接全量喂给 vllm，发挥其最大吞吐
        batch_size = None
        
    save_every_n_items = config['inference'].get('save_every_n_items', 10)
    test_limit = config['inference'].get('test_limit', 0)

    # 遍历数据集进行提取
    for dataset_name in datasets:
        input_path = os.path.join(data_root, dataset_name, input_subdir, input_filename)
        output_path = os.path.join(output_root, dataset_name, output_subdir, output_filename)

        if not os.path.exists(input_path):
            print(f"\n[Skip] {dataset_name}: 输入文件不存在 -> {input_path}")
            continue

        # 处理 test_limit
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
            extractor=extractor,
            pm=pm,
            batch_size=batch_size,
            save_every_n_items=save_every_n_items,
        )

        # 清理临时测试文件
        if test_limit and test_limit > 0 and os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n[All Done] 所有配置数据集提取流程处理完毕。")

if __name__ == '__main__':
    main()
