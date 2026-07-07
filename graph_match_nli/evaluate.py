"""
NLI-Graph 融合模型评估与批量推理脚本
==================================
功能：
  - 加载已训练的 NLI-Graph 融合模型
  - 批量对配置中的所有数据集（或指定单个数据集）进行推理
  - 将预测结果 pred_label 与 pred_prob 追加保存到各数据集的 our_results 子目录下
  - 后续的评分与指标计算应交给 evaluation 模块处理

用法：
  python graph_match_nli/evaluate.py
  python graph_match_nli/evaluate.py --dataset minicheck
  python graph_match_nli/evaluate.py --ckpt models/graph_match/best_f1.pt
"""

import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import NLIGraphClassifier
from dataset import NLIGraphDataset
from nli_labels import nli_logits_to_support_preds, nli_logits_to_support_probs
from utils.io_utils import load_yaml_config

load_config = load_yaml_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None,
                   help="配置文件路径。默认使用 configs/graph_match_nli.yaml")
    p.add_argument('--ckpt', default=None,
                   help="检查点文件路径。默认使用 config 中的 best_f1_path")
    p.add_argument('--dataset', default=None,
                   help="指定仅对某个数据集进行推理（如 minicheck）")
    return p.parse_args()


def run_chunked_inference(model, test_ds, device, label_spec, chunk_size: int, chunk_batch_size: int = 8):
    """长文档分块推理：doc 按句切成多个 chunk，每个 chunk 与 claim 独立跑一次模型，
    取所有 chunk 中的最大支持概率作为该样本的最终预测（对齐 MiniCheck 的 chunk + max 策略）。
    图（claim/doc 三元组图）不切分，所有 chunk 共享同一张图。
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="  推理(分块)", leave=False):
            pairs, label = test_ds.get_chunk_samples(idx, chunk_size=chunk_size)

            chunk_probs = []
            for i in range(0, len(pairs), chunk_batch_size):
                batch = Batch.from_data_list(
                    pairs[i:i + chunk_batch_size], follow_batch=['x_s', 'x_t']
                ).to(device)
                logits = model(batch.input_ids, batch.attention_mask, batch.token_type_ids, batch)
                probs = nli_logits_to_support_probs(logits, label_spec)
                chunk_probs.extend(probs.cpu().tolist())

            final_prob = max(chunk_probs) if chunk_probs else 0.0
            final_pred = 1 if final_prob > 0.5 else 0

            all_preds.append(final_pred)
            all_probs.append(final_prob)
            all_labels.append(label)

    return all_preds, all_probs, all_labels


def evaluate():
    args = parse_args()
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(base_dir, 'configs', 'graph_match_nli.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config      = load_config(config_path)['nli_graph']
    cfg_dir     = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    data_root = resolve(config['data'].get('data_root', '../data'))
    output_filename = config['data'].get('output_filename', 'qwen3.5_2b_tk.parquet')

    nli_model_path = resolve(config['model']['nli_model_path'])
    emb_model_path = resolve(config['model']['embedding_model_path'])
    
    if args.ckpt:
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else resolve(args.ckpt)
    else:
        ckpt_path = resolve(config['training']['best_f1_path'])
        
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

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

    model = NLIGraphClassifier(
        nli_model_path    = nli_model_path,
        node_input_dim    = config['model']['node_input_dim'],
        edge_input_dim    = config['model']['node_input_dim'],
        node_hidden_dim   = config['model']['node_hidden_dim'],
        num_prop_layers   = config['model']['num_prop_layers'],
        inject_layer_k    = config['model']['inject_layer_k'],
        num_heads         = config['model']['num_heads'],
        dropout           = config['model']['dropout'],
        freeze_nli_layers = config['model'].get('freeze_nli_layers', 0),
        num_labels        = config['model'].get('num_labels', 2),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.float()
    model.eval()
    label_spec = model.label_spec
    print(f"模型加载完毕（来自 Epoch {ckpt['epoch']}，Val F1={ckpt['val_f1']:.4f}）")
    print(
        f"  NLI 标签映射: entailment={label_spec.entailment_id}, "
        f"hallucination={label_spec.hallucination_id}, num_labels={label_spec.num_labels}"
    )

    # ---- 数据集列表 ----
    datasets = config['data'].get('datasets', [])
    if args.dataset:
        datasets = [args.dataset]

    print(f"开始批量推理，共 {len(datasets)} 个数据集...")
    test_batch_size = config['training'].get(
        'test_batch_size',
        config['training'].get('val_batch_size', config['training']['batch_size']),
    )
    print(f"  test_batch_size={test_batch_size}")

    # 长文档分块推理配置（不影响训练，仅在此处生效）：
    #   chunked=True 时，doc 按句切成多个 ~chunk_size token 的块，
    #   每块独立打分后取最大支持概率（对齐 MiniCheck 的 chunk + max 聚合策略）
    infer_cfg  = config.get('inference', {})
    chunked           = infer_cfg.get('chunked', False)
    chunk_size        = infer_cfg.get('chunk_size', 400)
    chunk_batch_size  = infer_cfg.get('chunk_batch_size', 8)
    if chunked:
        print(f"  [分块推理] 已启用: chunk_size={chunk_size}, chunk_batch_size={chunk_batch_size}")

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"[Dataset] {dataset_name}")
        
        # 确定输入路径
        if dataset_name.lower() == 'minicheck':
            test_path = resolve(config['data']['test_parquet'])
        else:
            test_path = os.path.join(data_root, dataset_name, 'data_with_graph', 'gemma_26b_tk.parquet')

        if not os.path.exists(test_path):
            print(f"  [Skip] 输入文件不存在 -> {test_path}")
            continue

        output_path = os.path.join(data_root, dataset_name, 'our_results', output_filename)
        print(f"  输入路径: {test_path}")
        print(f"  输出路径: {output_path}")

        # 构建测试集（分块模式下不需要预先缓存整篇 doc 的编码，节省内存与时间）
        test_ds = NLIGraphDataset(
            test_path, tokenizer, emb_model_path,
            max_length=config['model']['max_length'],
            device=str(device),
            embed_cache_path=None,  # 自动寻址
            preload_to_memory=not chunked,
        )

        if chunked:
            all_preds, all_probs, all_labels = run_chunked_inference(
                model, test_ds, device, label_spec,
                chunk_size=chunk_size, chunk_batch_size=chunk_batch_size,
            )
        else:
            test_loader = DataLoader(
                test_ds, batch_size=test_batch_size,
                shuffle=False, follow_batch=['x_s', 'x_t']
            )

            all_preds, all_probs, all_labels = [], [], []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"  推理 {dataset_name}", leave=False):
                    batch = batch.to(device)
                    logits = model(
                        batch.input_ids,
                        batch.attention_mask,
                        batch.token_type_ids,
                        batch
                    )
                    # 数据集 label 1=支持, 0=幻觉；由 label_spec 统一解码 NLI logits
                    probs = nli_logits_to_support_probs(logits, label_spec)
                    preds = nli_logits_to_support_preds(logits, label_spec)
                    labels = batch.y.view(-1).long()

                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_probs.extend(probs.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

        # 写出预测结果
        out_df = test_ds.df.copy()
        out_df['pred_label'] = all_preds
        out_df['pred_prob'] = all_probs
        
        # 只保留 id, claim, doc, label, pred_label 和 pred_prob
        cols_to_keep = ['id', 'claim', 'doc', 'label', 'pred_label', 'pred_prob']
        cols_to_keep = [col for col in cols_to_keep if col in out_df.columns]
        out_df = out_df[cols_to_keep]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_df.to_parquet(output_path, index=False)
        
        # 计算并打印基本指标
        acc = accuracy_score(all_labels, all_preds)
        bacc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        print(f"  [Done] 处理完成。样本数: {len(all_preds)} | Acc: {acc:.4f} | BAcc: {bacc:.4f} | F1: {f1:.4f}")
        print(f"  结果已写出: {output_path}")

    print("\n[All Done] 所有数据集批量推理完毕。")


if __name__ == '__main__':
    evaluate()
