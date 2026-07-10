"""
纯文本 NLI 批量推理（消融 w/o graph）

用法:
  python ablation/nli_text/evaluate.py
  python ablation/nli_text/evaluate.py --dataset SCIFACT
  python ablation/nli_text/evaluate.py --ckpt models/nli_text_ft/best_f1.pt
"""
import os
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NLITextDataset
from graph_match_nli.nli_labels import (
    nli_logits_to_support_preds,
    nli_logits_to_support_probs,
    resolve_nli_label_spec,
)
from utils.io_utils import load_yaml_config


def load_config(path: str) -> dict:
    return load_yaml_config(path)['ablation']['nli_ft']


def _collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'token_type_ids': torch.stack([b['token_type_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def run_chunked_inference(model, test_ds, device, label_spec, chunk_size, chunk_batch_size):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="  推理(分块)", leave=False):
            items, label = test_ds.get_chunk_batch_items(idx, chunk_size=chunk_size)
            chunk_probs = []
            for i in range(0, len(items), chunk_batch_size):
                batch_items = items[i:i + chunk_batch_size]
                batch = {
                    'input_ids': torch.stack([x['input_ids'] for x in batch_items]).to(device),
                    'attention_mask': torch.stack([x['attention_mask'] for x in batch_items]).to(device),
                    'token_type_ids': torch.stack([x['token_type_ids'] for x in batch_items]).to(device),
                }
                logits = model(**batch).logits
                probs = nli_logits_to_support_probs(logits, label_spec)
                chunk_probs.extend(probs.cpu().tolist())

            final_prob = max(chunk_probs) if chunk_probs else 0.0
            all_preds.append(1 if final_prob > 0.5 else 0)
            all_probs.append(final_prob)
            all_labels.append(label)

    return all_preds, all_probs, all_labels


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    base_dir = _PROJ_ROOT
    config_path = args.config or os.path.join(base_dir, 'configs', 'ablation.yaml')
    config = load_config(config_path)
    cfg_dir = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    data_root = resolve(config['data'].get('data_root', '../data'))
    output_filename = config['data'].get('output_filename', 'deberta-nli-text-ft.parquet')
    nli_model_path = resolve(config['model']['nli_model_path'])
    if not os.path.isdir(nli_model_path):
        raise FileNotFoundError(f"NLI 模型目录不存在: {nli_model_path}")

    ckpt_path = args.ckpt if args.ckpt else resolve(config['training']['best_f1_path'])
    if not os.path.isabs(ckpt_path) and args.ckpt:
        ckpt_path = resolve(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    _dev = config['model']['device']
    if _dev == 'cuda' and not torch.cuda.is_available():
        _dev = 'cpu'
    device = torch.device(_dev)

    print(f"设备: {device}")
    print(f"加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    label_spec = resolve_nli_label_spec(model.config.id2label)
    print(
        f"Epoch {ckpt.get('epoch')} | Val F1={ckpt.get('val_f1', 0):.4f} | "
        f"entailment={label_spec.entailment_id}, hallucination={label_spec.hallucination_id}"
    )

    datasets = config['data'].get('datasets', [])
    if args.dataset:
        datasets = [args.dataset]

    infer_cfg = config.get('inference', {})
    chunked = infer_cfg.get('chunked', True)
    chunk_size = infer_cfg.get('chunk_size', 400)
    chunk_batch_size = infer_cfg.get('chunk_batch_size', 8)
    test_batch_size = config['training'].get('test_batch_size', 24)

    if chunked:
        print(f"  [分块推理] chunk_size={chunk_size}, chunk_batch_size={chunk_batch_size}")
    else:
        print(f"  test_batch_size={test_batch_size}")

    for dataset_name in datasets:
        print(f"\n{'='*60}\n[Dataset] {dataset_name}")
        if dataset_name.lower() == 'minicheck':
            test_path = resolve(config['data']['test_parquet'])
        else:
            test_path = os.path.join(data_root, dataset_name, 'data_with_graph', 'gemma_26b_tk.parquet')

        if not os.path.exists(test_path):
            print(f"  [Skip] 不存在: {test_path}")
            continue

        output_path = os.path.join(data_root, dataset_name, 'ablation_results', output_filename)
        print(f"  输入: {test_path}")
        print(f"  输出: {output_path}")

        test_ds = NLITextDataset(
            test_path, tokenizer,
            max_length=config['model']['max_length'],
            preload_to_memory=not chunked,
        )

        if chunked:
            all_preds, all_probs, all_labels = run_chunked_inference(
                model, test_ds, device, label_spec, chunk_size, chunk_batch_size,
            )
        else:
            loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, collate_fn=_collate)
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc="  推理", leave=False):
                    logits = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device),
                    ).logits
                    probs = nli_logits_to_support_probs(logits, label_spec)
                    preds = nli_logits_to_support_preds(logits, label_spec)
                    all_preds.extend(preds.cpu().tolist())
                    all_probs.extend(probs.cpu().tolist())
                    all_labels.extend(batch['labels'].cpu().tolist())

        out_df = test_ds.df.copy()
        out_df['pred_label'] = all_preds
        out_df['pred_prob'] = all_probs
        cols = [c for c in ['id', 'claim', 'doc', 'label', 'pred_label', 'pred_prob'] if c in out_df.columns]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_df[cols].to_parquet(output_path, index=False)

        acc = accuracy_score(all_labels, all_preds)
        bacc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"  [Done] N={len(all_preds)} | Acc={acc:.4f} | BAcc={bacc:.4f} | F1={f1:.4f}")

    print("\n[All Done]")


if __name__ == '__main__':
    evaluate()
