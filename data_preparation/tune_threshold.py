"""
基于验证集（stage2_augment_v2/val.parquet）挑选 BAcc 最优的判决阈值，
再无偏地套用到 held-out 测试集上，验证阈值校准是否能带来真实收益。

用法：
  python data_preparation/tune_threshold.py --config configs/graph_match_nli_stage2_v2.yaml \
      --ckpt ../models/graph_match/roberta_large_mnli/best_f1_stage2_v2.pt
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'graph_match_nli'))

from model import NLIGraphClassifier
from dataset import NLIGraphDataset
from nli_labels import nli_logits_to_support_probs
from utils.io_utils import load_yaml_config
from evaluate import run_chunked_inference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dir = os.path.join(BASE_DIR, 'configs')
    config = load_yaml_config(os.path.join(cfg_dir, os.path.basename(args.config)))['nli_graph']

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nli_model_path = resolve(config['model']['nli_model_path'])
    emb_model_path = resolve(config['model']['embedding_model_path'])
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else resolve(args.ckpt)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)

    model = NLIGraphClassifier(
        nli_model_path=nli_model_path,
        node_input_dim=config['model']['node_input_dim'],
        edge_input_dim=config['model']['node_input_dim'],
        node_hidden_dim=config['model']['node_hidden_dim'],
        num_prop_layers=config['model']['num_prop_layers'],
        inject_layer_k=config['model']['inject_layer_k'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        freeze_nli_layers=config['model'].get('freeze_nli_layers', 0),
        num_labels=config['model'].get('num_labels', 2),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.float().eval()
    label_spec = model.label_spec
    print(f"已加载 checkpoint: {ckpt_path} (Epoch {ckpt.get('epoch')}, Val F1={ckpt.get('val_f1', 0):.4f})")

    val_path = resolve(config['data']['val_parquet'])
    val_ds = NLIGraphDataset(
        val_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device), embed_cache_path=None, preload_to_memory=False,
    )
    infer_cfg = config.get('inference', {})
    chunk_size = infer_cfg.get('chunk_size', 400)
    chunk_batch_size = infer_cfg.get('chunk_batch_size', 8)

    print(f"在验证集（{len(val_ds)} 条）上跑推理以校准阈值...")
    _, val_probs, val_labels = run_chunked_inference(
        model, val_ds, device, label_spec,
        chunk_size=chunk_size, chunk_batch_size=chunk_batch_size,
    )
    val_df = val_ds.df.copy()
    val_df['pred_prob'] = val_probs
    val_df['label'] = val_labels

    # 全局阈值（所有验证样本合并求解，跨 3 个数据集共享同一个阈值，避免过拟合到单个小数据集）
    y = np.array(val_labels)
    p = np.array(val_probs)
    best_bacc, best_th = 0.0, 0.5
    for th in np.arange(0.05, 0.96, 0.01):
        pred = (p > th).astype(int)
        b = balanced_accuracy_score(y, pred)
        if b > best_bacc:
            best_bacc, best_th = b, th
    print(f"\n[验证集校准结果] 全局最优阈值={best_th:.2f} | 验证集BAcc: 0.5阈值={balanced_accuracy_score(y, (p>0.5).astype(int)):.4f} -> 校准后={best_bacc:.4f}")

    # 按数据集分别求最优阈值（不同数据集 label 分布/校准特性不同，共用一个全局阈值会互相拖累）
    per_ds_th = {}
    print()
    for ds_name, grp in val_df.groupby('source_dataset'):
        yy, pp = grp['label'].values, grp['pred_prob'].values
        b0 = balanced_accuracy_score(yy, (pp > 0.5).astype(int))
        d_best_bacc, d_best_th = 0.0, 0.5
        for th in np.arange(0.05, 0.96, 0.01):
            b = balanced_accuracy_score(yy, (pp > th).astype(int))
            if b > d_best_bacc:
                d_best_bacc, d_best_th = b, th
        per_ds_th[ds_name] = d_best_th
        print(f"  {ds_name:16s} n={len(grp):4d} | val BAcc(th=0.5)={b0:.4f} -> val BAcc(th={d_best_th:.2f})={d_best_bacc:.4f}")

    # 用该阈值无偏地套到 held-out 测试集
    print(f"\n[Held-out 测试集验证 - 全局阈值 {best_th:.2f}]（在验证集上选出，测试集完全没参与选择）：")
    targets = ['AggreFact-CNN', 'Reveal', 'ExpertQA']
    output_filename = config['data'].get('output_filename')
    for ds in targets:
        result_path = os.path.join(BASE_DIR, 'data', ds, 'our_results', output_filename)
        if not os.path.exists(result_path):
            print(f"  [Skip] {ds}: 找不到 {result_path}")
            continue
        df = pd.read_parquet(result_path)
        yy, pp = df['label'].values, df['pred_prob'].values
        b0 = balanced_accuracy_score(yy, (pp > 0.5).astype(int))
        b1 = balanced_accuracy_score(yy, (pp > best_th).astype(int))
        acc1 = accuracy_score(yy, (pp > best_th).astype(int))
        f1_1 = f1_score(yy, (pp > best_th).astype(int), zero_division=0)
        print(f"  {ds:16s} n={len(df):4d} | test BAcc(th=0.5)={b0:.4f} -> test BAcc(th={best_th:.2f})={b1:.4f} | Acc={acc1:.4f} F1={f1_1:.4f}")

    print(f"\n[Held-out 测试集验证 - 按数据集独立阈值]（每个数据集用自己 val 集选出的阈值）：")
    for ds in targets:
        result_path = os.path.join(BASE_DIR, 'data', ds, 'our_results', output_filename)
        if not os.path.exists(result_path) or ds not in per_ds_th:
            continue
        th = per_ds_th[ds]
        df = pd.read_parquet(result_path)
        yy, pp = df['label'].values, df['pred_prob'].values
        b0 = balanced_accuracy_score(yy, (pp > 0.5).astype(int))
        b1 = balanced_accuracy_score(yy, (pp > th).astype(int))
        acc1 = accuracy_score(yy, (pp > th).astype(int))
        f1_1 = f1_score(yy, (pp > th).astype(int), zero_division=0)
        print(f"  {ds:16s} n={len(df):4d} | 阈值={th:.2f} | test BAcc(th=0.5)={b0:.4f} -> test BAcc(校准后)={b1:.4f} | Acc={acc1:.4f} F1={f1_1:.4f}")


if __name__ == '__main__':
    main()
