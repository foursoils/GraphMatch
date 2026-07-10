"""纯文本 NLI 微调数据集：仅 (doc, claim, label)，不使用图结构。"""
import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from utils.dataset_utils import split_doc_into_chunks
from utils.path_utils import is_rank0, log_rank0


class NLITextDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512,
                 preload_to_memory: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        log_rank0(f"  [NLITextDataset] {parquet_path} | N={len(self.df)}")

        self.preload_to_memory = preload_to_memory
        self.cached_samples = None
        if preload_to_memory:
            from tqdm import tqdm
            log_rank0(f"  [NLITextDataset] 缓存 {len(self.df)} 条样本...")
            self.cached_samples = [
                self._encode_row(self.df.iloc[i]) for i in tqdm(
                    range(len(self.df)), desc="Caching", disable=not is_rank0()
                )
            ]

    def _tokenize(self, doc_text: str, claim_text: str):
        encoding = self.tokenizer(
            doc_text,
            claim_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        if 'token_type_ids' in encoding:
            token_type_ids = encoding['token_type_ids'].squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids

    def _encode_row(self, row):
        label = int(row['label'])
        input_ids, attention_mask, token_type_ids = self._tokenize(
            str(row.get('doc', '')), str(row.get('claim', ''))
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label, dtype=torch.long),
        }

    def get_chunk_batch_items(self, idx, chunk_size: int = 400):
        """长文档分块推理：返回多个 (encoding dict) 与 label。"""
        row = self.df.iloc[idx]
        label = int(row['label'])
        claim_text = str(row.get('claim', ''))
        chunks = split_doc_into_chunks(str(row.get('doc', '')), self.tokenizer, chunk_size=chunk_size)

        items = []
        for chunk_text in chunks:
            input_ids, attention_mask, token_type_ids = self._tokenize(chunk_text, claim_text)
            items.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            })
        return items, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cached_samples is not None:
            return self.cached_samples[idx]
        return self._encode_row(self.df.iloc[idx])
