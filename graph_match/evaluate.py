"""graph_match 批量评估脚本。"""

import os
import sys
import warnings

warnings.filterwarnings('ignore', message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings('ignore', message="An issue occurred while importing 'torch-sparse'")

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from graph_match.common import maybe_relaunch_multigpu

maybe_relaunch_multigpu('infer')

import argparse

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from graph_match.common import load_config, parse_binary_pred
from graph_match.dataset import GraphMatchDataset, graph_collate_fn
from graph_match.model import LLMGraphModel
from utils.path_utils import configure_dist_process_logging, is_rank0, log_rank0, resolve_num_workers, resolve_path


def _verify_lora_loaded(model, lora_dir: str) -> bool:
    from peft import PeftModel
    from safetensors import safe_open

    if not isinstance(model.llm, PeftModel):
        return False
    adapter_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    if not os.path.exists(adapter_path):
        return False
    with safe_open(adapter_path, framework='pt') as f:
        saved_keys = [k for k in f.keys() if k.endswith('.lora_A.weight')]
        if not saved_keys:
            return False
        saved_key, saved = saved_keys[0], f.get_tensor(saved_keys[0])
    suffix = saved_key.split('layers.', 1)[-1] if 'layers.' in saved_key else saved_key
    model_keys = [
        k for k in model.llm.state_dict()
        if k.endswith(suffix) or k.endswith(suffix.replace('.weight', '.default.weight'))
    ]
    if not model_keys:
        return False
    return torch.allclose(model.llm.state_dict()[model_keys[0]].cpu().float(), saved.float(), atol=1e-5)


def load_checkpoint(model: LLMGraphModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.gmn.load_state_dict(ckpt['gmn'])
    model.projector.load_state_dict(ckpt['projector'])
    model.cross_attn_layer.load_state_dict(ckpt['cross_attn'])
    if 'macro_bias' in ckpt:
        model.macro_bias.load_state_dict(ckpt['macro_bias'])
    elif 'graph_global_proj' in ckpt:
        log_rank0('[Warn] 旧版 Macro 检查点不兼容，Macro 将使用随机初始化。')

    if is_rank0():
        log_rank0(f"[Ckpt] Epoch={ckpt.get('epoch', '?')}, val_BAcc={ckpt.get('val_bacc', '?')}")

    lora_dir = os.path.join(os.path.dirname(ckpt_path), 'lora_adapter')
    if not os.path.isdir(lora_dir):
        return
    from peft import PeftModel
    if isinstance(model.llm, PeftModel):
        raise RuntimeError('model.llm 已是 PeftModel，请用 apply_lora=False 初始化。')
    model.llm = PeftModel.from_pretrained(model.llm, lora_dir, is_trainable=False)
    if is_rank0():
        ok = _verify_lora_loaded(model, lora_dir)
        log_rank0(f"[Ckpt] LoRA: {lora_dir} ({'校验通过' if ok else '校验失败'})")


@torch.no_grad()
def evaluate_dataset(model, parquet_path, embed_path, model_cfg, infer_cfg,
                     output_path, accelerator, test_limit=0, embed_cache_path=None):
    ds = GraphMatchDataset(
        parquet_path=parquet_path,
        embed_model_path=embed_path,
        is_train=False,
        embed_cache_path=embed_cache_path,
    )
    if test_limit > 0:
        ds = Subset(ds, list(range(min(test_limit, len(ds)))))

    loader = DataLoader(
        ds,
        batch_size=infer_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=resolve_num_workers(infer_cfg.get('num_workers', 2)),
        collate_fn=graph_collate_fn,
    )
    model, loader = accelerator.prepare(model, loader)

    all_indices, all_preds, all_labels = [], [], []
    model.eval()
    for batch in tqdm(loader, desc='  推理', leave=False, disable=not accelerator.is_main_process):
        result = accelerator.unwrap_model(model).inference(batch)
        device = accelerator.device
        all_preds.extend(accelerator.gather_for_metrics(torch.tensor(
            [parse_binary_pred(p) for p in result['pred']], dtype=torch.long, device=device
        )).cpu().tolist())
        all_labels.extend(accelerator.gather_for_metrics(torch.tensor(
            [int(l) for l in result['label']], dtype=torch.long, device=device
        )).cpu().tolist())
        all_indices.extend(accelerator.gather_for_metrics(torch.tensor(
            [int(i) for i in batch['index']], dtype=torch.long, device=device
        )).cpu().tolist())

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None

    unique = {}
    for idx, pred, label in zip(all_indices, all_preds, all_labels):
        unique.setdefault(idx, (pred, label))

    underlying = ds.dataset if isinstance(ds, Subset) else ds
    out_df = underlying.df.copy()
    if test_limit > 0:
        out_df = out_df.iloc[:min(test_limit, len(out_df))].copy()

    preds, labels = [], []
    for idx in range(len(out_df)):
        if idx in unique:
            p, l = unique[idx]
            preds.append(p)
            labels.append(l)
        else:
            preds.append(-1)
            labels.append(-1)

    out_df['pred_label'] = preds
    cols = [c for c in ('id', 'claim', 'doc', 'label', 'pred_label') if c in out_df.columns]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df[cols].to_parquet(output_path, index=False)

    valid_p = [p for p in preds if p != -1]
    valid_l = [l for p, l in zip(preds, labels) if p != -1]
    return {
        'Acc': accuracy_score(valid_l, valid_p) if valid_p else 0.0,
        'BAcc': balanced_accuracy_score(valid_l, valid_p) if valid_p else 0.0,
        'F1': f1_score(valid_l, valid_p, average='binary', zero_division=0) if valid_p else 0.0,
        'n_samples': len(preds),
        'valid_samples': len(valid_p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/graph_match.yaml')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--dataset', default=None, help='只评估指定数据集')
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    data_cfg, model_cfg, train_cfg, infer_cfg = (
        config['data'], config['model'], config['training'], config['infer']
    )

    gpu_ids = infer_cfg.get('gpu_ids')
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    configure_dist_process_logging()

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    output_dir = resolve_path(train_cfg['output_dir'])
    ckpt_path = args.ckpt or os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'检查点不存在: {ckpt_path}')

    if accelerator.is_main_process:
        print('[Init] 初始化模型...')
    model = LLMGraphModel(config, device=accelerator.device, apply_lora=False)
    load_checkpoint(model, ckpt_path)
    model.eval()

    embed_path = resolve_path(model_cfg['embed_model_path'])
    data_root = resolve_path(data_cfg['data_root'])
    out_fname = data_cfg.get('output_filename', 'gemma3_1b.parquet')
    datasets = [args.dataset] if args.dataset else data_cfg.get('datasets', [])

    for ds_name in datasets:
        if accelerator.is_main_process:
            print(f"\n{'=' * 50}\n数据集: {ds_name}")

        if ds_name == 'minicheck':
            parquet_path = resolve_path(data_cfg['val_file'])
        else:
            parquet_path = os.path.join(data_root, ds_name, 'data_with_graph', 'gemma_26b_tk.parquet')

        if not os.path.exists(parquet_path):
            if accelerator.is_main_process:
                print(f'  [Skip] 文件不存在: {parquet_path}')
            continue

        output_path = os.path.join(data_root, ds_name, 'our_results', out_fname)
        if accelerator.is_main_process:
            print(f'  输入: {parquet_path}\n  输出: {output_path}')

        metrics = evaluate_dataset(
            model, parquet_path, embed_path, model_cfg, infer_cfg,
            output_path, accelerator, infer_cfg.get('test_limit', 0),
        )
        if accelerator.is_main_process and metrics:
            print(f"  [Done] n={metrics['n_samples']} | Acc={metrics['Acc']:.4f} | "
                  f"BAcc={metrics['BAcc']:.4f} | F1={metrics['F1']:.4f}")

    if accelerator.is_main_process:
        print('\n[All Done]')


if __name__ == '__main__':
    main()
