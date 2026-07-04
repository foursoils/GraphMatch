"""
GraphCheck 消融实验 - 推理程序
================================
功能：
  - 加载 ablation.yaml 中 check 节点的配置
  - 读取各数据集下 data_with_graph/gemma_26b_tk.parquet（与 kg 消融实验一致）
  - 从检查点还原 GNN + Projector 权重
  - 批量推理，解析 0/1 标签后保存为 parquet 文件
  - 支持断点续跑（基于 id 列跳过已处理数据）

用法：
  python check_infer.py
  python check_infer.py --config ../../configs/ablation.yaml
  python check_infer.py --config ../../configs/ablation.yaml --ckpt_path /path/to/best_model.pt
"""

import os
import re
import json
import argparse

import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.graphcheck import GraphCheck


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PRED_COL     = 'pred_label'
INPUT_FILE   = 'gemma_26b_tk.parquet'   # 与 kg 消融实验保持一致
GRAPH_SUBDIR = 'data_with_graph'


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """加载 YAML，返回 ablation.check 节点。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['ablation']['check']


def resolve_path(base_dir: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    cleaned = rel_or_abs.lstrip('.').lstrip('/').lstrip('\\')
    return os.path.normpath(os.path.join(base_dir, cleaned))


def parse_binary_label(raw: str) -> int:
    """
    从模型生成文本中提取 0/1 标签（与 kg 模块策略保持一致）。
    取最后一个出现的 \\b[01]\\b；匹配失败返回 2（解析失败）。
    """
    if not raw:
        return 2
    matches = re.findall(r'\b([01])\b', raw.strip())
    return int(matches[-1]) if matches else 2


def load_checkpoint(model: GraphCheck, ckpt_path: str):
    """从检查点文件还原 GNN 和 Projector 的权重。"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.graph_encoder.load_state_dict(ckpt['gnn_state_dict'])
    model.projector.load_state_dict(ckpt['proj_state_dict'])
    print(f"  [Ckpt] 加载检查点 (Epoch {ckpt.get('epoch', '?')}): {ckpt_path}")
    return model


def save_results(records: list, output_path: str):
    """将新记录追加到已有 parquet 文件，不存在则新建。"""
    new_df = pd.DataFrame(records)
    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        pd.concat([existing_df, new_df], ignore_index=True).to_parquet(output_path, index=False)
    else:
        new_df.to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# 单数据集推理（纯文本 batch，通过 DataLoader 外部分批）
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    model: GraphCheck,
    config: dict,
):
    """对单个数据集执行推理并保存结果（parquet）。"""
    print(f"\n{'='*60}")
    print(f"[Dataset] {dataset_name}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)

    # 断点续跑
    processed_ids: set = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            if 'id' in existing_df.columns:
                processed_ids = set(existing_df['id'])
                print(f"  [Resume] 已处理 {len(processed_ids)} 条，跳过")
        except Exception as e:
            print(f"  [Warning] 读取已有输出失败，从头开始: {e}")

    df_todo = df[~df['id'].isin(processed_ids)].copy()
    if len(df_todo) == 0:
        print("  [Done] 所有数据已处理完毕。")
        return

    # ---- 逐行构建 KG Batch 进行推理 ----
    # 使用自定义的 GraphCheckDataset 进行在线图转换
    from dataset import GraphCheckDataset, graphcheck_collate_fn

    # GraphCheckDataset 接受路径，因为我们是在对已有文件做处理，
    # 稍微改装一下：其实我们可以直接把 actual_input 作为路径传给 Dataset。
    embed_path = resolve_path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              config['model']['embed_model_path'])
    infer_dataset = GraphCheckDataset(input_path, embed_path, device=str(model.device))
    
    # 覆盖 Dataset 里的 df
    infer_dataset.df = df_todo

    batch_size    = config['infer'].get('batch_size', 8)
    num_workers   = config['infer'].get('num_workers', 4)

    # 动态根据是否使用预计算特征决定 num_workers
    infer_num_workers = num_workers if infer_dataset.use_precomputed else 0
    print(f"  [Init] DataLoader num_workers -> {infer_num_workers}")

    loader        = DataLoader(infer_dataset, batch_size=batch_size,
                               shuffle=False, drop_last=False,
                               pin_memory=True, num_workers=infer_num_workers,
                               collate_fn=graphcheck_collate_fn)

    parse_fail_count  = 0
    records_to_save   = []

    model.eval()
    for batch in tqdm(loader, desc=f"Inferring {dataset_name}"):
        with torch.no_grad():
            output = model.inference(batch)

        for idx, (sample_id, pred_text) in enumerate(zip(output['id'], output['pred'])):
            label = parse_binary_label(pred_text)
            if label == 2:
                parse_fail_count += 1
            # 还原原始行数据
            orig_row = df_todo[df_todo['id'] == sample_id].iloc[0].to_dict()
            orig_row[PRED_COL] = label
            records_to_save.append(orig_row)

    save_results(records_to_save, output_path)
    print(f"  [Done] 共 {len(records_to_save)} 条，解析失败 {parse_fail_count} 条。")
    print(f"  [Done] 结果保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_inference(local_rank, world_size, config, ckpt_path_override=None):
    import torch.distributed as dist

    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        dist.init_process_group(backend=backend, init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ---- 数据集配置 ----
    datasets     = config['data']['datasets']
    data_root    = resolve_path(base_dir, config['data']['data_root'])
    output_root  = resolve_path(base_dir, config['data']['output_root'])
    output_filename = config['data'].get('output_filename', 'graphcheck_infer.parquet')
    test_limit   = config['infer'].get('test_limit', 0)

    # 分配数据集
    if is_dist:
        my_datasets = [datasets[i] for i in range(len(datasets)) if i % world_size == local_rank]
        print(f"[Rank {local_rank}] 负责推理的数据集: {my_datasets}")
    else:
        my_datasets = datasets

    # ---- 检查点路径 ----
    ckpt_dir  = resolve_path(base_dir, config['training']['output_dir'])
    ckpt_path = ckpt_path_override or os.path.join(ckpt_dir, 'best_model.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"检查点文件不存在: {ckpt_path}\n"
            "请先运行 check_train.py 进行训练，或通过 --ckpt_path 指定正确路径。"
        )

    # ---- 初始化模型并加载检查点 ----
    if local_rank == 0:
        print("[Init] 初始化 GraphCheck 模型 ...")
    model = GraphCheck(config)
    model = load_checkpoint(model, ckpt_path)
    if local_rank == 0:
        print("[Init] 模型加载完毕。")

    # ---- 遍历数据集 ----
    for dataset_name in my_datasets:
        input_path  = os.path.join(data_root, dataset_name, GRAPH_SUBDIR, INPUT_FILE)
        output_path = os.path.join(output_root, dataset_name, 'check_results', output_filename)

        if not os.path.exists(input_path):
            print(f"\n[Skip] {dataset_name}: 输入文件不存在 -> {input_path}")
            continue

        # 测试模式：只处理前 N 条
        actual_input = input_path
        if test_limit and test_limit > 0:
            df_full = pd.read_parquet(input_path)
            df_full = df_full.head(test_limit).copy()
            tmp_path = input_path.replace('.parquet', '_test_tmp.parquet')
            df_full.to_parquet(tmp_path, index=False)
            actual_input = tmp_path
            print(f"[Data] 测试模式：{dataset_name} 仅处理前 {test_limit} 条")

        process_dataset(
            dataset_name=dataset_name,
            input_path=actual_input,
            output_path=output_path,
            model=model,
            config=config,
        )

        if test_limit and test_limit > 0 and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()

    if local_rank == 0:
        print("\n[All Done] 所有数据集推理完毕。")


def main_worker(local_rank, world_size, config_path, ckpt_path_override):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    # 重新加载 config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(config_path)
    run_inference(local_rank, world_size, config, ckpt_path_override)


def main():
    parser = argparse.ArgumentParser(description="GraphCheck Ablation: Inference")
    parser.add_argument('--config',    type=str, default=None, help="配置文件路径")
    parser.add_argument('--ckpt_path', type=str, default=None, help="检查点路径（覆盖 YAML 配置）")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    config_path = args.config or os.path.join(base_dir, 'configs', 'ablation.yaml')
    config      = load_config(config_path)

    tensor_parallel_size = config['infer'].get('tensor_parallel_size', 1)
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if tensor_parallel_size > 1 and not is_torchrun:
        import torch.multiprocessing as mp
        print(f"[Init] 检测到配置 tensor_parallel_size={tensor_parallel_size}，正在自动拉起多卡 DDP 分布式推理...")
        mp.spawn(main_worker, nprocs=tensor_parallel_size, args=(tensor_parallel_size, config_path, args.ckpt_path))
    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        run_inference(local_rank, world_size, config, args.ckpt_path)


if __name__ == '__main__':
    main()
