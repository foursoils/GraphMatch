"""graph_match 数据集：parquet → PairData batch。"""

import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from utils.dataset_utils import build_pair_data, load_precomputed_embeddings
from utils.path_utils import log_rank0, resolve_path

from graph_match.common import load_prompt

_DEFAULT_SYSTEM = (
    "You are an expert fact-checker. "
    "Given a document and a claim, reason step by step and determine "
    "whether the document supports the claim."
)
_DEFAULT_USER = "<doc>\n{{doc}}\n</doc>\n\n<claim>\n{{claim}}\n</claim>"


class GraphMatchDataset(Dataset):
    """读取带图 parquet，构建 instruction / target / graph_pair。"""

    def __init__(
        self,
        parquet_path: str,
        embed_model_path: str,
        is_train: bool = True,
        train_target: str = 'answer_only',
        embed_cache_path: str = None,
    ):
        super().__init__()
        self.is_train = is_train
        self.train_target = train_target

        log_rank0(f"[Dataset] 加载数据: {parquet_path}")
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

        self.system_prompt = load_prompt(
            'prompts/hallu_detect/system_prompt.txt', _DEFAULT_SYSTEM
        )
        self.user_prompt_template = load_prompt(
            'prompts/hallu_detect/user_prompt.txt', _DEFAULT_USER
        )

        self.embeddings_dict, self.use_precomputed, self.embeddings_path = (
            load_precomputed_embeddings(
                parquet_path=parquet_path,
                embed_model_path=embed_model_path,
                embed_cache_path=embed_cache_path,
            )
        )
        if not self.use_precomputed:
            raise FileNotFoundError(
                f"找不到预计算 embedding: {self.embeddings_path}。"
                f"请检查 data.train_embed_file / val_embed_file。"
            )

    def __len__(self):
        return len(self.df)

    def _build_instruction(self, doc: str, claim: str) -> str:
        return (
            self.user_prompt_template
            .replace('{{doc}}', doc)
            .replace('{{claim}}', claim)
        )

    def _build_target(self, label: int, cot: str = '') -> str:
        answer = '1' if label == 1 else '0'
        if self.train_target == 'cot_and_answer' and cot:
            return f"{cot.strip()}\n{answer}"
        return answer

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc = str(row.get('doc', ''))
        claim = str(row.get('claim', ''))
        label = int(row.get('label', 0))
        sample_id = row['id']

        item = {
            'index': idx,
            'id': sample_id,
            'instruction': self._build_instruction(doc, claim),
            'label': label,
            'graph_pair': build_pair_data(
                claim_graph_str=str(row.get('graph_claim', '')),
                doc_graph_str=str(row.get('graph_doc', '')),
                sample_id=sample_id,
                embeddings_dict=self.embeddings_dict,
                use_precomputed=True,
            ),
        }
        if self.is_train:
            cot = str(row.get('gt_trial', '')) if self.train_target == 'cot_and_answer' else ''
            item['target'] = self._build_target(label, cot)
        return item


def graph_collate_fn(batch):
    out = {
        'id': [x['id'] for x in batch],
        'instruction': [x['instruction'] for x in batch],
        'label': [x['label'] for x in batch],
        'index': [x['index'] for x in batch],
        'graph_pair': Batch.from_data_list(
            [x['graph_pair'] for x in batch],
            follow_batch=['x_s', 'x_t'],
        ),
    }
    if 'target' in batch[0]:
        out['target'] = [x['target'] for x in batch]
    return out

# 兼容旧名
LLMGraphDataset = GraphMatchDataset
llm_graph_collate_fn = graph_collate_fn
