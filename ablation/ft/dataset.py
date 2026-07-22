"""纯文本幻觉检测 Dataset（无图）。"""

import pandas as pd
from torch.utils.data import Dataset

from ablation.ft.common import load_prompt
from utils.path_utils import log_rank0

_DEFAULT_SYSTEM = (
    "You are an expert fact-checking assistant specialized in hallucination detection.\n"
    "Return ONLY a single integer: 1 if supported, 0 otherwise."
)
_DEFAULT_USER = "<doc>\n{{doc}}\n</doc>\n\n<claim>\n{{claim}}\n</claim>"


class HalluTextDataset(Dataset):
    """读取 processed_data parquet，构建 instruction / target。"""

    def __init__(
        self,
        parquet_path: str,
        system_prompt_path: str = 'prompts/hallu_detect/system_prompt.txt',
        user_prompt_path: str = 'prompts/hallu_detect/user_prompt.txt',
        is_train: bool = True,
    ):
        super().__init__()
        self.is_train = is_train
        log_rank0(f"[Dataset] 加载数据: {parquet_path}")
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.system_prompt = load_prompt(system_prompt_path, _DEFAULT_SYSTEM)
        self.user_prompt_template = load_prompt(user_prompt_path, _DEFAULT_USER)

    def __len__(self):
        return len(self.df)

    def _build_instruction(self, doc: str, claim: str) -> str:
        return (
            self.user_prompt_template
            .replace('{{doc}}', doc)
            .replace('{{claim}}', claim)
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc = str(row.get('doc', ''))
        claim = str(row.get('claim', ''))
        label = int(row.get('label', 0))
        item = {
            'index': idx,
            'id': row['id'] if 'id' in row else idx,
            'instruction': self._build_instruction(doc, claim),
            'label': label,
            'system_prompt': self.system_prompt,
        }
        if self.is_train:
            item['target'] = '1' if label == 1 else '0'
        return item


def text_collate_fn(batch):
    out = {
        'id': [x['id'] for x in batch],
        'instruction': [x['instruction'] for x in batch],
        'label': [x['label'] for x in batch],
        'index': [x['index'] for x in batch],
        'system_prompt': batch[0]['system_prompt'],
    }
    if 'target' in batch[0]:
        out['target'] = [x['target'] for x in batch]
    return out
