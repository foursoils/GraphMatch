"""
NLI-Graph 融合模型评估脚本

用法:
  python -m nli_graph.evaluate                                  # 默认 best_loss.pt + 配置中的 test_parquet
  python -m nli_graph.evaluate --ckpt best_f1.pt                # 切换检查点
  python -m nli_graph.evaluate --test-parquet data/xxx.parquet  # 切换数据集（绝对路径或相对项目根的路径）
  python -m nli_graph.evaluate --ckpt best_f1.pt --tag f1_run   # 输出文件名带后缀
"""
import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import NLIGraphClassifier
from dataset import NLIGraphDataset


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None,
                   help="检查点文件路径。默认使用 config 中的 best_loss_path")
    p.add_argument('--test-parquet', default=None,
                   help="覆盖 config 里的测试集路径；可填绝对路径或相对项目根的路径")
    p.add_argument('--tag', default=None,
                   help="预测结果文件名后缀，默认根据 ckpt 自动推断")
    return p.parse_args()


def evaluate():
    args = parse_args()
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'graph_match_nli.yaml')
    config      = load_config(config_path)['nli_graph']
    cfg_dir     = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    # 测试集：CLI 优先，其次 config
    # 相对路径解析顺序：① 相对当前 cwd（用户最直观）② 相对 cfg_dir（与 config 一致）③ 相对 base_dir
    if args.test_parquet:
        if os.path.isabs(args.test_parquet):
            test_path = args.test_parquet
        else:
            candidates = [
                os.path.abspath(args.test_parquet),
                os.path.normpath(os.path.join(cfg_dir, args.test_parquet)),
                os.path.normpath(os.path.join(base_dir, args.test_parquet)),
            ]
            test_path = next((c for c in candidates if os.path.exists(c)), candidates[0])
    else:
        test_path = resolve(config['data']['test_parquet'])

    nli_model_path = resolve(config['model']['nli_model_path'])
    emb_model_path = resolve(config['model']['embedding_model_path'])
    if args.ckpt:
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else resolve(args.ckpt)
    else:
        ckpt_path = resolve(config['training']['best_loss_path'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # 输出文件名 tag 推断
    tag = args.tag or os.path.splitext(os.path.basename(ckpt_path))[0]

    _dev = config['model']['device']
    if _dev == 'cuda' and not torch.cuda.is_available():
        _dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif _dev == 'mps' and not torch.backends.mps.is_available():
        _dev = 'cpu'
    device = torch.device(_dev)

    print(f"使用设备: {device}")
    print(f"加载检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)

    print("构建测试集...")
    test_ds = NLIGraphDataset(
        test_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device),
        embed_cache_path=resolve(config['data']['test_embed_file']) if config['data'].get('test_embed_file') else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['training']['batch_size'],
        shuffle=False, follow_batch=['x_s', 'x_t']
    )

    model = NLIGraphClassifier(
        nli_model_path    = nli_model_path,
        node_input_dim    = config['model']['node_input_dim'],
        edge_input_dim    = config['model']['node_input_dim'],
        node_hidden_dim   = config['model']['node_hidden_dim'],
        num_prop_layers   = config['model']['num_prop_layers'],
        inject_layer_k    = config['model']['inject_layer_k'],
        num_heads         = config['model']['num_heads'],
        dropout           = config['model']['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.float()
    model.eval()
    print(f"模型加载完毕（来自 Epoch {ckpt['epoch']}，Val F1={ckpt['val_f1']:.4f}）")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(
                batch.input_ids,
                batch.attention_mask,
                batch.token_type_ids,
                batch
            )
            # NLI class 0=entailment(支持), class 1=not_entailment(幻觉)
            # dataset label 1=支持, 0=幻觉 → 与 NLI class 方向相反
            probs = torch.softmax(logits, dim=-1)[:, 0]          # entailment(支持) 概率
            preds = (logits[:, 0] > logits[:, 1]).long()         # 支持(1) vs 幻觉(0)
            labels = batch.y.squeeze(-1).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # 指标输出
    acc   = accuracy_score(all_labels, all_preds)
    bacc  = balanced_accuracy_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec  = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec   = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    print("\n========== 测试集评估结果 ==========")
    print(f"  Accuracy          : {acc:.4f}   (普通准确率，受类别比例影响)")
    print(f"  Balanced Accuracy : {bacc:.4f}   (平衡准确率，对不平衡更公平)")
    print(f"  F1  (binary)      : {f1:.4f}   (正类F1，以支持=1为正类)")
    print(f"  F1  (macro)       : {f1_macro:.4f}   (两类F1均值，不受比例影响)")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  AUC-ROC           : {auc:.4f}   (排序能力，越接近1越好)")
    print("\n--- 详细分类报告 ---")
    print(classification_report(
        all_labels, all_preds,
        target_names=['幻觉(0)', '支持(1)'],
        zero_division=0
    ))

    # 写出预测结果
    out_df = test_ds.df.copy()
    out_df['pred_label'] = all_preds
    out_df['pred_prob']  = all_probs
    out_path = os.path.join(os.path.dirname(test_path), f'nli_graph_predictions_{tag}.parquet')
    out_df.to_parquet(out_path, index=False)
    print(f"\n预测结果已写出: {out_path}")
    print(f"使用检查点: {ckpt_path}")
    print(f"使用测试集: {test_path}")


if __name__ == '__main__':
    evaluate()
