"""
GraphCheck 消融实验 - 数据集处理模块
====================================
负责：
  1. 读取 Parquet 文件中的 JSON 字符串三元组
  2. 调用指定的 Embedding 模型（例如 Qwen3-Embedding），将节点和边转换为特征张量
"""

import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer


def get_embedding_path(parquet_path: str, embed_model_path: str) -> str:
    """
    根据 parquet 路径和 embed 模型路径自动计算预计算特征 .pt 文件的集中存储路径。
    支持 Windows 和 Linux 路径格式。
    """
    model_name = os.path.basename(embed_model_path)
    
    # 统一并规范化路径
    norm_path = os.path.normpath(parquet_path)
    parts = norm_path.split(os.sep)
    
    # 查找 'data' 目录的索引
    try:
        data_idx = parts.index('data')
        base_parts = parts[:data_idx+1]
        sub_parts = parts[data_idx+1:]
    except ValueError:
        return None
        
    if not sub_parts:
        return None
        
    # 判断是否是 minicheck 的特殊结构
    # data/minicheck/data_with_graph/generator/split.parquet
    if len(sub_parts) >= 4 and sub_parts[0] == 'minicheck':
        generator = sub_parts[2]
        split_file = sub_parts[3]
        split_name = os.path.splitext(split_file)[0]
        new_parts = base_parts + ['embeddings', model_name, 'minicheck', generator, f"{split_name}.pt"]
    # 常规数据集结构
    # data/dataset_name/data_with_graph/generator.parquet
    elif len(sub_parts) >= 3:
        dataset_name = sub_parts[0]
        gen_file = sub_parts[2]
        gen_name = os.path.splitext(gen_file)[0]
        new_parts = base_parts + ['embeddings', model_name, dataset_name, f"{gen_name}.pt"]
    else:
        return None
        
    return os.path.normpath(os.path.sep.join(new_parts))


def textualize_graph(graph_str: str):
    """
    将 JSON 格式的图字符串解析为节点和边。
    返回: nodes_df (含 node_id, node_attr), edges_df (含 src, edge_attr, dst)
    """
    if not graph_str or not isinstance(graph_str, str):
        return pd.DataFrame(columns=['node_attr', 'node_id']), pd.DataFrame(columns=['src', 'edge_attr', 'dst'])
    
    try:
        triples = json.loads(graph_str)
    except Exception:
        triples = []

    if not triples:
        return pd.DataFrame(columns=['node_attr', 'node_id']), pd.DataFrame(columns=['src', 'edge_attr', 'dst'])

    nodes_dict = {}
    edges_list = []

    for tri in triples:
        if len(tri) != 3: continue
        src, edge_attr, dst = tri

        src = src.lower().strip() if src else " "
        edge_attr = edge_attr.lower().strip() if edge_attr else " "
        dst = dst.lower().strip() if dst else " "

        if src not in nodes_dict:
            nodes_dict[src] = len(nodes_dict)
        if dst not in nodes_dict:
            nodes_dict[dst] = len(nodes_dict)

        edges_list.append({
            'src': nodes_dict[src],
            'edge_attr': edge_attr,
            'dst': nodes_dict[dst]
        })

    nodes_df = pd.DataFrame(nodes_dict.items(), columns=['node_attr', 'node_id'])
    edges_df = pd.DataFrame(edges_list)
    return nodes_df, edges_df


class GraphCheckDataset(Dataset):
    """
    读取 Parquet 并构建图的 Dataset。
    若存在预计算的 Embedding 文件，将直接读取以加速，并且无需在 GPU 运行 Embedding 模型。
    """
    def __init__(self, parquet_path: str, embed_model_path: str, device: str = 'cuda'):
        super().__init__()
        print(f"[Dataset] 正在加载数据: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        self.device = device
        
        # 1. 尝试查找预计算的 Embedding 文件
        self.embeddings_path = get_embedding_path(parquet_path, embed_model_path)
        self.use_precomputed = False
        
        if self.embeddings_path and os.path.exists(self.embeddings_path):
            try:
                print(f"[Dataset] 发现预计算 Embedding 文件，启动缓存加速模式:\n          {self.embeddings_path}")
                self.embeddings_dict = torch.load(self.embeddings_path, map_location='cpu')
                self.use_precomputed = True
            except Exception as e:
                print(f"[Dataset] [Warning] 读取已存在 Embedding 文件失败，回退到在线计算模式: {e}")

        # 2. 如果没有找到缓存特征，才在 GPU 加载原始 Embedding 模型进行在线推理
        if not self.use_precomputed:
            print(f"[Dataset] 未检测到匹配的预计算特征文件，正在加载 Embedding 模型进行在线计算: {embed_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(embed_model_path, trust_remote_code=True)
            self.embed_model = AutoModel.from_pretrained(embed_model_path, trust_remote_code=True).to(self.device).eval()

    def __len__(self):
        return len(self.df)

    @torch.no_grad()
    def get_text_embedding(self, texts: list):
        """调用 Embedding 模型将文本列表转换为 Tensor"""
        if not texts:
            return torch.zeros((0, self.embed_model.config.hidden_size), device=self.device)
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.embed_model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        embeddings = torch.sum(last_hidden * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings.cpu()

    def build_pyg_data(self, graph_str: str, sample_id: any = None, graph_key: str = 'claim') -> Data:
        """
        构建 PyG Data 对象。若有缓存特征直接查字典，否则在线提取。
        """
        nodes, edges = textualize_graph(graph_str)
        
        # A. 使用预计算的向量特征
        if self.use_precomputed and sample_id in self.embeddings_dict:
            features = self.embeddings_dict[sample_id]
            # 读取并将 float16 转回 float32，防止后续 GNN 网络训练类型报错
            x = features[f'{graph_key}_x'].float()
            e = features[f'{graph_key}_e'].float()
        else:
            # B. 在线实时计算逻辑
            if len(nodes) == 0:
                hidden_dim = self.embed_model.config.hidden_size
                return Data(x=torch.zeros((1, hidden_dim)), 
                            edge_index=torch.zeros((2, 0), dtype=torch.long), 
                            edge_attr=torch.zeros((0, hidden_dim)), 
                            num_nodes=1)
            
            x = self.get_text_embedding(nodes['node_attr'].tolist())
            e = self.get_text_embedding(edges['edge_attr'].tolist())

        edge_index = torch.tensor([edges['src'].tolist(), edges['dst'].tolist()], dtype=torch.long)
        num_nodes = len(nodes) if len(nodes) > 0 else 1
        return Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=num_nodes)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        # 构造 prompt 中需要的 text
        doc_text = str(row['doc'])
        claim_text = str(row['claim'])
        text = f"Document: {doc_text}\nClaim: {claim_text}\nDoes the document support the claim?"
        
        label_text = ""
        if 'label' in row:
            label_val = row['label']
            if str(label_val) == '1': label_text = "Yes"
            elif str(label_val) == '0': label_text = "No"
            else: label_text = str(label_val)
            
        return {
            'id': sample_id,
            'text': text,
            'label': label_text,
            'claim_kg': self.build_pyg_data(row.get('graph_claim', ''), sample_id, 'claim'),
            'doc_kg': self.build_pyg_data(row.get('graph_doc', ''), sample_id, 'doc')
        }


def graphcheck_collate_fn(batch):
    """
    DataLoader 的拼装函数。
    PyG 的 Data 需要用 Batch.from_data_list 拼装，普通 list 正常保留。
    """
    ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    claim_kgs = [item['claim_kg'] for item in batch]
    doc_kgs = [item['doc_kg'] for item in batch]
    
    return {
        'id': ids,
        'text': texts,
        'label': labels,
        'claim_kg': Batch.from_data_list(claim_kgs),
        'doc_kg': Batch.from_data_list(doc_kgs)
    }
