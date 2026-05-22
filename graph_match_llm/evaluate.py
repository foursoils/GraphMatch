"""
graph_match_llm - 评估脚本
============================
功能：
  - 加载已训练的检查点（GNN + Projector + Cross-Attn）以及 LoRA adapter
  - 在各数据集上批量推理，解析 Yes/No，计算 BAcc / F1 / AUC 等指标
  - 结果保存为 parquet（与其他对比实验格式一致）

用法：
  cd /root/workspace/GraphMatch_code
  python -m graph_match_llm.evaluate
  python -m graph_match_llm.evaluate --config configs/graph_match_llm.yaml --ckpt models/llm_graph/best_model.pt
  python -m graph_match_llm.evaluate --dataset minicheck  # 只跑单个数据集
"""

import os
import sys
import re
import json
import argparse

import torch
import yaml
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from graph_match_llm.dataset import LLMGraphDataset, llm_graph_collate_fn
from graph_match_llm.model   import LLMGraphModel


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['llm_graph']


def resolve(base: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    cleaned = rel.lstrip('.').lstrip('/').lstrip('\\')
    return os.path.normpath(os.path.join(base, cleaned))


def parse_binary_pred(text: str) -> int:
    """从生成文本中解析 Yes/No 或 1/0，返回 1/0/-1（-1 表示解析失败）。"""
    if not text:
        return -1
    text_cleaned = text.strip().lower()
    
    # 优先精确匹配单个数字 0 或 1
    if text_cleaned == '1':
        return 1
    if text_cleaned == '0':
        return 0
        
    # 精确匹配 yes/no
    if text_cleaned == 'yes':
        return 1
    if text_cleaned == 'no':
        return 0

    # 优先找 "answer is: yes/no"
    m = re.search(r'answer\s+is\s*:\s*(yes|no)', text_cleaned)
    if m:
        return 1 if m.group(1) == 'yes' else 0

    # 其次找 "answer is: 1/0"
    m = re.search(r'answer\s+is\s*:\s*(1|0)', text_cleaned)
    if m:
        return 1 if m.group(1) == '1' else 0

    # 退而求其次：找最后一个 yes/no 或 1/0
    matches = re.findall(r'\b(yes|no|1|0)\b', text_cleaned)
    if matches:
        last_match = matches[-1]
        if last_match in ('yes', '1'):
            return 1
        elif last_match in ('no', '0'):
            return 0

    return -1


def compute_metrics(preds: list, labels: list) -> dict:
    """计算 BAcc / Accuracy / F1 / Precision / Recall。"""
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        if p == 1 and l == 1:   tp += 1
        elif p == 1 and l == 0: fp += 1
        elif p == 0 and l == 0: tn += 1
        else:                   fn += 1

    total    = tp + fp + tn + fn
    acc      = (tp + tn) / total if total > 0 else 0.0
    prec     = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall   = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    spec     = tn / (tn + fp)   if (tn + fp) > 0 else 0.0
    f1       = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    bacc     = (recall + spec) / 2

    return {
        'BAcc': round(bacc * 100, 2),
        'Acc':  round(acc  * 100, 2),
        'F1':   round(f1   * 100, 2),
        'P':    round(prec * 100, 2),
        'R':    round(recall * 100, 2),
    }


def load_checkpoint(model: LLMGraphModel, ckpt_path: str):
    """加载 GNN / Projector / Cross-Attn 权重，以及 LoRA adapter（如有）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.gmn.load_state_dict(ckpt['gmn'])
    model.projector.load_state_dict(ckpt['projector'])
    model.cross_attn_layer.load_state_dict(ckpt['cross_attn'])
    if 'gmn_cls_head' in ckpt:
        model.gmn_cls_head.load_state_dict(ckpt['gmn_cls_head'])
    epoch = ckpt.get('epoch', '?')
    bacc  = ckpt.get('val_bacc', '?')
    print(f"[Ckpt] 加载检查点 (Epoch={epoch}, val_BAcc={bacc})")

    # 尝试加载 LoRA adapter
    lora_dir = os.path.join(os.path.dirname(ckpt_path), 'lora_adapter')
    if os.path.isdir(lora_dir):
        try:
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
            print(f"[Ckpt] LoRA adapter 已加载: {lora_dir}")
        except Exception as e:
            print(f"[Warn] LoRA adapter 加载失败: {e}")


# ---------------------------------------------------------------------------
# 单数据集评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model:      LLMGraphModel,
    parquet_path: str,
    embed_path:   str,
    model_cfg:    dict,
    infer_cfg:    dict,
    output_path:  str,
    test_limit:   int = 0,
) -> dict:
    ds = LLMGraphDataset(
        parquet_path=parquet_path,
        embed_model_path=embed_path,
        tokenizer=model.tokenizer,
        max_txt_len=model_cfg.get('max_txt_len', 1024),
        is_train=False,
        device=f"cuda:{model.device_id}",
    )

    if test_limit > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(test_limit, len(ds)))))

    loader = DataLoader(
        ds,
        batch_size=infer_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=infer_cfg.get('num_workers', 2),
        collate_fn=llm_graph_collate_fn,
    )

    records = []
    all_preds, all_labels = [], []

    model.eval()
    for batch in tqdm(loader, desc=f"  推理", leave=False):
        result = model.inference(batch)
        for sid, pred_text, label in zip(result['id'], result['pred'], result['label']):
            p = parse_binary_pred(pred_text)
            all_preds.append(p)
            all_labels.append(int(label))
            records.append({
                'id':         sid,
                'pred_label': p,
                'pred_text':  pred_text,
                'label':      int(label),
            })

    # 过滤解析失败样本后计算指标
    valid_p = [p for p in all_preds if p != -1]
    valid_l = [l for p, l in zip(all_preds, all_labels) if p != -1]
    parse_rate = len(valid_p) / max(len(all_preds), 1)
    metrics = compute_metrics(valid_p, valid_l) if valid_p else {}
    metrics['parse_rate'] = round(parse_rate * 100, 2)
    metrics['n_samples']  = len(all_preds)

    # 保存结果
    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    return metrics


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='configs/graph_match_llm.yaml')
    parser.add_argument('--ckpt',    default=None, help='检查点路径（默认 output_dir/best_model.pt）')
    parser.add_argument('--dataset', default=None, help='只评估指定数据集（留空=全部）')
    args = parser.parse_args()

    config_path = os.path.join(_PROJ_ROOT, args.config) if not os.path.isabs(args.config) \
                  else args.config
    config    = load_config(config_path)
    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    infer_cfg = config['infer']

    embed_path  = resolve(_PROJ_ROOT, model_cfg['embed_model_path'])
    output_dir  = resolve(_PROJ_ROOT, train_cfg['output_dir'])
    data_root   = resolve(_PROJ_ROOT, data_cfg['data_root'])
    out_fname   = data_cfg.get('output_filename', 'llm_graph_pred.parquet')
    test_limit  = infer_cfg.get('test_limit', 0)

    ckpt_path = args.ckpt or os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # ---- 初始化模型 ----
    print("[Init] 初始化模型...")
    model = LLMGraphModel(config)
    load_checkpoint(model, ckpt_path)
    model.eval()

    # ---- 数据集列表 ----
    datasets = data_cfg.get('datasets', [])
    if args.dataset:
        datasets = [args.dataset]

    # minicheck 特殊处理（test split 路径格式不同）
    minicheck_test = resolve(_PROJ_ROOT, data_cfg.get(
        'val_file', '../data/minicheck/data_with_graph/gemma_26b_tk/val.parquet'
    ))

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"数据集: {ds_name}")

        if ds_name == 'minicheck':
            parquet_path = minicheck_test
        else:
            parquet_path = os.path.join(data_root, ds_name, 'data_with_graph', 'gemma_26b_tk.parquet')

        if not os.path.exists(parquet_path):
            print(f"  [Skip] 文件不存在: {parquet_path}")
            continue

        output_path = os.path.join(data_root, ds_name, 'ablation_results', out_fname)

        metrics = evaluate_dataset(
            model=model,
            parquet_path=parquet_path,
            embed_path=embed_path,
            model_cfg=model_cfg,
            infer_cfg=infer_cfg,
            output_path=output_path,
            test_limit=test_limit,
        )
        all_results[ds_name] = metrics

        print(f"  BAcc={metrics.get('BAcc','?')}  Acc={metrics.get('Acc','?')}  "
              f"F1={metrics.get('F1','?')}  parse={metrics.get('parse_rate','?')}%  "
              f"n={metrics.get('n_samples','?')}")
        print(f"  结果已保存: {output_path}")

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print("全部评估结果汇总：")
    print(f"{'数据集':<20} {'BAcc':>7} {'Acc':>7} {'F1':>7} {'parse%':>8}")
    print('-' * 50)
    baccs = []
    for ds_name, m in all_results.items():
        bacc = m.get('BAcc', 0)
        baccs.append(bacc)
        print(f"{ds_name:<20} {bacc:>7.2f} {m.get('Acc',0):>7.2f} "
              f"{m.get('F1',0):>7.2f} {m.get('parse_rate',0):>8.2f}")
    if baccs:
        print(f"\n平均 BAcc (across {len(baccs)} datasets): {sum(baccs)/len(baccs):.2f}")

    # 保存汇总 JSON
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == '__main__':
    main()
