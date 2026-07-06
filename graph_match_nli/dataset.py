import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import Dataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from utils.dataset_utils import PairData, build_pair_data, load_precomputed_embeddings
from utils.path_utils import is_rank0, log_rank0

class NLIGraphDataset(Dataset):
    def __init__(self,
                 parquet_path: str,
                 tokenizer,
                 embedding_model_path: str,
                 max_length: int = 512,
                 device: str = 'cpu',
                 emb_dim: int = 1024,
                 embed_cache_path: str = None,
                 preload_to_memory: bool = True):
        """
        :param parquet_path:         含 doc / claim / graph_claim / subgraph_doc / label 的 Parquet
        :param tokenizer:            DeBERTa tokenizer（外部传入，避免多次加载）
        :param embedding_model_path: SentenceTransformer 路径（用于图节点 Embedding）
        :param max_length:           DeBERTa 最大序列长度
        :param device:               Embedding 编码设备
        :param emb_dim:              节点 Embedding 维度（Qwen3-Embedding-0.6B=1024）
        :param embed_cache_path:     预计算 Embedding 路径 (.pt)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emb_dim = emb_dim
        self.device = device

        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

        # ── 自动适配 doc 图列名 ───────────────────────────────────────────
        if 'subgraph_doc' in self.df.columns:
            self.doc_col = 'subgraph_doc'
        elif 'graph_doc' in self.df.columns:
            self.doc_col = 'graph_doc'
        else:
            raise ValueError("数据中既无 'subgraph_doc' 也无 'graph_doc' 列")
        log_rank0(f"  使用 doc 图列: {self.doc_col}")

        # ── 优先载入预计算 Embedding ─────────────────────────────────────
        self.embeddings_dict, self.use_precomputed, self.embeddings_path = load_precomputed_embeddings(
            parquet_path=parquet_path,
            embed_model_path=embedding_model_path,
            embed_cache_path=embed_cache_path
        )

        # ── 批量构建 Embedding 缓存 (在线计算 fallback) ────────────────────
        if not self.use_precomputed:
            from sentence_transformers import SentenceTransformer
            log_rank0("收集唯一文本，批量编码节点 Embedding...")
            
            # 预解析三元组以提取所有文本
            claim_triplets = [self._parse(r['graph_claim']) for _, r in self.df.iterrows()]
            doc_triplets   = [self._parse(r[self.doc_col]) for _, r in self.df.iterrows()]
            
            all_texts = set()
            for trips in claim_triplets + doc_triplets:
                for tri in trips:
                    all_texts.update(tri)
            all_texts = list(all_texts)
            log_rank0(f"共 {len(all_texts)} 个唯一文本，开始编码...")

            emb_model = SentenceTransformer(embedding_model_path, device=device)
            embeddings = emb_model.encode(
                all_texts, batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=is_rank0()
            )
            self.text_emb_cache = {t: emb for t, emb in zip(all_texts, embeddings)}
            
            # 释放显存
            del emb_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_rank0("节点 Embedding 缓存构建完毕！")

        self.preload_to_memory = preload_to_memory
        if self.preload_to_memory:
            log_rank0(f"  [Dataset] 正在预处理并缓存 {len(self.df)} 个样本到内存中...")
            self.cached_samples = []
            from tqdm import tqdm
            for idx in tqdm(range(len(self.df)), desc="Caching samples", disable=not is_rank0()):
                self.cached_samples.append(self._get_raw_item(idx))
            log_rank0("  [Dataset] 预处理缓存完成！")

    @staticmethod
    def _parse(json_str) -> list:
        if not json_str:
            return []
        try:
            result = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(result, list) and all(
                isinstance(t, list) and len(t) == 3 for t in result
            ):
                return result
        except Exception:
            pass
        return []

    def __len__(self):
        return len(self.df)

    def _get_raw_item(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        label = int(row['label'])

        # ── DeBERTa 文本输入 ─────────────────────────────────────────────
        doc_text   = str(row.get('doc',   ''))
        claim_text = str(row.get('claim', ''))

        encoding = self.tokenizer(
            doc_text,
            claim_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids      = encoding['input_ids'].squeeze(0)        # [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)   # [seq_len]
        if 'token_type_ids' in encoding:
            token_type_ids = encoding['token_type_ids'].squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)

        # ── 图对 ────────────────────────────────────────────────────────
        pair = build_pair_data(
            claim_graph_str=str(row.get('graph_claim', '')),
            doc_graph_str=str(row.get(self.doc_col, '')),
            sample_id=sample_id,
            embeddings_dict=getattr(self, 'embeddings_dict', None),
            use_precomputed=self.use_precomputed,
            text_emb_cache=getattr(self, 'text_emb_cache', None),
            device=self.device
        )
        
        # 统一使用公共的 PairData 并打包 NLI 附加特征
        pair.y = torch.tensor([label], dtype=torch.float32)
        pair.input_ids = input_ids
        pair.attention_mask = attention_mask
        pair.token_type_ids = token_type_ids
        
        return pair

    def __getitem__(self, idx):
        if self.preload_to_memory:
            return self.cached_samples[idx]
        return self._get_raw_item(idx)
