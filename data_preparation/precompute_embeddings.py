"""
GraphCheck 消融实验 - 集中式图节点/边文本嵌入预计算脚本
======================================================
功能：
  - 从 configs/data_prep.yaml 读取配置
  - 加载指定的 Embedding 模型 (如 Qwen3-Embedding-0.6B)
  - 对常规数据集和 minicheck 分别收集图中所有唯一的节点/边文本进行去重
  - 大 Batch 一次性进行 Embedding 推理，并按样本 id 打包存成字典
  - 将 Embedding 结果以 float16 格式集中保存到 data/embeddings/<model_name> 下的对应目录中
"""

import os
import sys
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to sys.path so we can reuse the shared utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.path_utils import resolve_path as _resolve_path_from_root
from utils.io_utils import load_yaml_config
from utils.dataset_utils import textualize_graph, get_text_embedding_online as get_text_embeddings_batch


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    return load_yaml_config(config_path)


def resolve_path(base_dir: str, rel_or_abs: str) -> str:
    """解析路径（base_dir 已固定为项目根目录，故直接复用共享实现）。"""
    return _resolve_path_from_root(rel_or_abs)


def process_and_save(
    input_path: str,
    output_path: str,
    tokenizer,
    embed_model,
    device: str,
    batch_size: int,
    hidden_dim: int,
    dataset_desc: str
):
    """
    处理单个 parquet 数据文件，并保存其对应的 embedding 字典文件（.pt）。
    支持断点续跑：只计算未处理过的 id 的样本特征。
    """
    if not os.path.exists(input_path):
        print(f"  [Skip] 输入文件不存在: {input_path}")
        return

    print(f"\n[Processing] 开始处理: {dataset_desc}")
    print(f"  输入路径: {input_path}")
    print(f"  输出路径: {output_path}")

    # 1. 读取数据集
    df = pd.read_parquet(input_path)
    total_samples = len(df)

    # 2. 检查是否有已计算的特征文件
    embeddings_dict = {}
    if os.path.exists(output_path):
        try:
            embeddings_dict = torch.load(output_path, map_location='cpu')
            print(f"  [Resume] 找到已有的 Embedding 文件，包含 {len(embeddings_dict)} / {total_samples} 条样本。")
        except Exception as e:
            print(f"  [Warning] 读取已有 Embedding 文件失败，将重新计算整个数据集: {e}")
            embeddings_dict = {}

    # 3. 差集计算，找出需要处理的行
    processed_ids = set(embeddings_dict.keys())
    df_todo = df[~df['id'].isin(processed_ids)].copy()

    if len(df_todo) == 0:
        print("  [Done] 所有样本的 Embedding 已预计算完毕，跳过。")
        return

    print(f"  本次运行仍需计算: {len(df_todo)} / {total_samples} 条样本。")

    # 4. 收集所有的 node_attr 和 edge_attr
    unique_texts = set()
    print("  正在扫描增量图数据并收集唯一文本属性...")
    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Collecting texts"):
        # Claim KG
        claim_nodes, claim_edges = textualize_graph(row.get('graph_claim', ''))
        unique_texts.update(claim_nodes['node_attr'].tolist())
        unique_texts.update(claim_edges['edge_attr'].tolist())
        
        # Doc KG
        doc_nodes, doc_edges = textualize_graph(row.get('graph_doc', ''))
        unique_texts.update(doc_nodes['node_attr'].tolist())
        unique_texts.update(doc_edges['edge_attr'].tolist())

    unique_texts_list = list(unique_texts)
    print(f"  扫描完毕，增量唯一文本数: {len(unique_texts_list)}")

    # 5. 批量计算 Embedding
    text_to_emb = {}
    if unique_texts_list:
        print(f"  开始计算嵌入 (Batch Size: {batch_size}) ...")
        for i in tqdm(range(0, len(unique_texts_list), batch_size), desc="Computing embeddings"):
            batch_texts = unique_texts_list[i : i + batch_size]
            embeddings = get_text_embeddings_batch(batch_texts, tokenizer, embed_model, device)
            for text, emb in zip(batch_texts, embeddings):
                text_to_emb[text] = emb

    # 6. 组装新增样本的特征张量字典并转换成 float16
    print("  正在组装新增样本的节点和边特征张量...")
    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Assembling graphs"):
        sample_id = row['id']

        # Claim Graph
        claim_nodes, claim_edges = textualize_graph(row.get('graph_claim', ''))
        if len(claim_nodes) == 0:
            claim_x = torch.zeros((1, hidden_dim), dtype=torch.float16)
            claim_e = torch.zeros((0, hidden_dim), dtype=torch.float16)
        else:
            claim_x = torch.stack([text_to_emb[t] for t in claim_nodes['node_attr']]).half()
            claim_e = torch.stack([text_to_emb[t] for t in claim_edges['edge_attr']]).half() if len(claim_edges) > 0 \
                      else torch.zeros((0, hidden_dim), dtype=torch.float16)

        # Doc Graph
        doc_nodes, doc_edges = textualize_graph(row.get('graph_doc', ''))
        if len(doc_nodes) == 0:
            doc_x = torch.zeros((1, hidden_dim), dtype=torch.float16)
            doc_e = torch.zeros((0, hidden_dim), dtype=torch.float16)
        else:
            doc_x = torch.stack([text_to_emb[t] for t in doc_nodes['node_attr']]).half()
            doc_e = torch.stack([text_to_emb[t] for t in doc_edges['edge_attr']]).half() if len(doc_edges) > 0 \
                    else torch.zeros((0, hidden_dim), dtype=torch.float16)

        embeddings_dict[sample_id] = {
            'claim_x': claim_x,
            'claim_e': claim_e,
            'doc_x': doc_x,
            'doc_e': doc_e
        }

    # 7. 保存 .pt 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_dict, output_path)
    print(f"  [Done] 成功保存 Embedding 文件 (当前包含 {len(embeddings_dict)} 条样本) -> {output_path}")

    # 清理垃圾
    del df, df_todo, unique_texts, unique_texts_list, text_to_emb, embeddings_dict
    gc.collect()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'data_prep.yaml')
    
    # 1. 加载配置项
    config = load_config(config_path)['embedding_preparation']
    
    device = config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    batch_size = config.get('batch_size', 256)
    embed_model_path = resolve_path(base_dir, config['embed_model_path'])
    data_root = resolve_path(base_dir, config['data_root'])
    
    model_name = os.path.basename(embed_model_path)
    
    # 2. 初始化 Embedding 模型
    print(f"[Init] 正在加载 Embedding  tokenizer & model: {embed_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_path, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(embed_model_path, trust_remote_code=True).to(device).eval()
    hidden_dim = embed_model.config.hidden_size
    print(f"[Init] 模型加载成功。特征维度: {hidden_dim}，运行设备: {device}")

    # 3. 遍历 datasets 配置列表
    for dataset in config['datasets']:
        if dataset == 'minicheck':
            # 处理 minicheck 特殊数据集
            mc_cfg = config.get('minicheck', {})
            if not mc_cfg:
                print("  [Warning] datasets 中包含 minicheck，但未在配置中找到 minicheck 相关配置项，跳过。")
                continue
            
            mc_name = mc_cfg.get('name', 'minicheck')
            mc_generators = mc_cfg.get('graph_generators', config['graph_generators'])
            splits = mc_cfg.get('splits', ['train', 'val', 'test'])
            
            for generator in mc_generators:
                print(f"\n{'#'*70}\n[Minicheck Generator] 正在处理 minicheck 图生成器: {generator}\n{'#'*70}")
                for split in splits:
                    input_file = os.path.join(data_root, mc_name, "data_with_graph", generator, f"{split}.parquet")
                    output_file = os.path.join(data_root, "embeddings", model_name, mc_name, generator, f"{split}.pt")
                    process_and_save(
                        input_path=input_file,
                        output_path=output_file,
                        tokenizer=tokenizer,
                        embed_model=embed_model,
                        device=device,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        dataset_desc=f"{mc_name} / {generator} / {split}"
                    )
        else:
            # 处理常规数据集
            print(f"\n{'#'*70}\n[Dataset] 正在处理数据集: {dataset}\n{'#'*70}")
            for generator in config['graph_generators']:
                input_file = os.path.join(data_root, dataset, "data_with_graph", f"{generator}.parquet")
                output_file = os.path.join(data_root, "embeddings", model_name, dataset, f"{generator}.pt")
                process_and_save(
                    input_path=input_file,
                    output_path=output_file,
                    tokenizer=tokenizer,
                    embed_model=embed_model,
                    device=device,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    dataset_desc=f"{dataset} / {generator}"
                )

    print("\n✅ 所有选定数据集的 Embedding 预计算完成！")


if __name__ == '__main__':
    main()
