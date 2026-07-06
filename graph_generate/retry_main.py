import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path
from utils.prompt_utils import PromptManager
from utils.model_engine import LocalQwenExtractor
from utils.io_utils import load_yaml_config

load_config = load_yaml_config

def process_retry_dataset(
    dataset_name: str,
    target_path: str,
    extractor: LocalQwenExtractor,
    pm: PromptManager,
    batch_size: int,
    save_every_n_items: int,
    max_retries: int,
):
    """对单个数据集运行图谱提取重试逻辑。"""
    print(f"\n{'='*60}")
    print(f"[Dataset] {dataset_name}")
    print(f"  目标结果文件: {target_path}")

    if not os.path.exists(target_path):
        print(f"  [Skip] {dataset_name}: 结果文件不存在 -> {target_path}")
        return

    df = pd.read_parquet(target_path)
    
    # 检验必要的字段
    if 'claim_graph_status' not in df.columns or 'doc_graph_status' not in df.columns:
        print(f"  [Skip] {dataset_name}: 不存在 status 列，可能还未运行主提取程序")
        return

    # 循环重试提取
    for attempt in range(1, max_retries + 1):
        mask_claim = df['claim_graph_status'] == 0
        mask_doc = df['doc_graph_status'] == 0
        df_to_process = df[mask_claim | mask_doc].copy()
        
        if len(df_to_process) == 0:
            print("  [Done] 所有数据的图谱提取状态均正常，无需重试。")
            break
            
        total_tasks = sum(mask_claim) + sum(mask_doc)
        print(f"  [Attempt {attempt}/{max_retries}] 发现 {len(df_to_process)} 条记录存在提取失败，共计 {total_tasks} 个任务待重试...")
        
        jobs = []
        tasks_per_row = {}
        for idx, row in df_to_process.iterrows():
            tasks = 0
            if row['doc_graph_status'] == 0:
                jobs.append((idx, 'doc', pm.get_messages(str(row['doc']))))
                tasks += 1
            if row['claim_graph_status'] == 0:
                jobs.append((idx, 'claim', pm.get_messages(str(row['claim']))))
                tasks += 1
            tasks_per_row[idx] = tasks
                
        completed_items = 0
        pbar = tqdm(total=len(jobs), desc=f"  重试中 (Attempt {attempt})", unit="次任务")
        
        if getattr(extractor, 'engine_type', 'api') == 'api':
            def process_single_job(job):
                idx, target_type, msgs = job
                try:
                    res = extractor.client.chat.completions.create(
                        model=extractor.serving_model,
                        messages=msgs,
                        max_tokens=extractor.max_tokens,
                        temperature=extractor.temperature,
                        top_p=extractor.top_p,
                        presence_penalty=extractor.presence_penalty,
                        extra_body={
                            "top_k": extractor.top_k,
                            "min_p": extractor.min_p,
                            "repetition_penalty": extractor.repetition_penalty,
                            "enable_thinking": extractor.enable_thinking
                        }
                    )
                    content = res.choices[0].message.content
                except Exception as e:
                    print(f"\n[Warning] API 重试异常 (index={idx}, type={target_type}): {e}")
                    content = "FAILED"
                    
                status, res_parsed = extractor._parse_graph_tag(content)
                return idx, target_type, status, res_parsed
                
            max_workers = min(len(jobs), extractor.api_max_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {executor.submit(process_single_job, job): job for job in jobs}
                for future in concurrent.futures.as_completed(future_to_job):
                    idx, target_type, status, res_parsed = future.result()
                    
                    if target_type == 'doc':
                        df.at[idx, 'doc_graph_status'] = status
                        df.at[idx, 'graph_doc'] = json.dumps(res_parsed, ensure_ascii=False)
                    elif target_type == 'claim':
                        df.at[idx, 'claim_graph_status'] = status
                        df.at[idx, 'graph_claim'] = json.dumps(res_parsed, ensure_ascii=False)
                        
                    pbar.update(1)
                    
                    tasks_per_row[idx] -= 1
                    if tasks_per_row[idx] == 0:
                        completed_items += 1
                        if completed_items > 0 and completed_items % save_every_n_items == 0:
                            df.to_parquet(target_path, index=False)
                            pbar.set_postfix({"已落盘": completed_items})
        else:
            # vLLM / 批处理模式
            actual_batch_size = batch_size if batch_size is not None else (len(jobs) if jobs else 1)
            for i in range(0, len(jobs), actual_batch_size):
                batch_jobs = jobs[i:i+actual_batch_size]
                batch_msgs = [j[2] for j in batch_jobs]
                
                res_list = extractor.batch_extract(batch_msgs)
                
                for j, res in zip(batch_jobs, res_list):
                    idx, target_type, _ = j
                    status, res_parsed = extractor._parse_graph_tag(res)
                    
                    if target_type == 'doc':
                        df.at[idx, 'doc_graph_status'] = status
                        df.at[idx, 'graph_doc'] = json.dumps(res_parsed, ensure_ascii=False)
                    elif target_type == 'claim':
                        df.at[idx, 'claim_graph_status'] = status
                        df.at[idx, 'graph_claim'] = json.dumps(res_parsed, ensure_ascii=False)
                        
                    pbar.update(1)
                    
                    tasks_per_row[idx] -= 1
                    if tasks_per_row[idx] == 0:
                        completed_items += 1
                        if completed_items > 0 and completed_items % save_every_n_items == 0:
                            df.to_parquet(target_path, index=False)
                            pbar.set_postfix({"已落盘": completed_items})
                            
        df.to_parquet(target_path, index=False)
        pbar.close()

    mask_claim_after = df['claim_graph_status'] == 0
    mask_doc_after = df['doc_graph_status'] == 0
    failed_after = sum(mask_claim_after | mask_doc_after)
    print(f"  [Done] {dataset_name} 重试结束。仍失败记录数: {failed_after}")


def main():
    parser = argparse.ArgumentParser(description="Retry graph extraction tasks that failed parsing.")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="配置文件路径，默认使用 configs/graph_gen.yaml"
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        default=3,
        help="最大重试轮数"
    )
    args = parser.parse_args()

    if args.config:
        config_path = resolve_path(args.config)
    else:
        config_path = resolve_path("configs/graph_gen.yaml")
        
    print(f"[Init] 加载配置文件: {config_path}")
    config = load_config(config_path)['minicheck']
    
    # 提取配置参数
    output_root = resolve_path(config['data']['output_root'])
    output_subdir = config['data'].get('output_subdir', 'data_with_graph')
    output_filename = config['data'].get('output_filename', 'gemma_26b_tk.parquet')
    datasets = config['data']['datasets']

    prompts_dir = resolve_path("prompts/graph_gen")
    
    # 仅加载引擎和 Prompt 管理器一次
    extractor = LocalQwenExtractor(config)
    pm = PromptManager(prompts_dir)
    
    engine_type = config['inference'].get('engine_type', 'api').lower()
    if engine_type == 'api':
        batch_size = config['inference'].get('api', {}).get('batch_size', config['inference'].get('batch_size', 20))
    else:
        # vLLM 模式下，不需要在 Python 层切小 batch
        batch_size = None
        
    save_every_n_items = config['inference'].get('save_every_n_items', 10)
    
    # 遍历数据集重试
    for dataset_name in datasets:
        target_path = os.path.join(output_root, dataset_name, output_subdir, output_filename)
        process_retry_dataset(
            dataset_name=dataset_name,
            target_path=target_path,
            extractor=extractor,
            pm=pm,
            batch_size=batch_size,
            save_every_n_items=save_every_n_items,
            max_retries=args.max_retries,
        )

    print("\n[All Done] 所有数据集重试流程执行完毕。")

if __name__ == '__main__':
    main()
