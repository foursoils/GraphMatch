"""
纯文本 NLI 微调（消融 w/o graph）

在 MoritzLaurer 等 NLI 预训练 DeBERTa 上，仅用 (doc, claim) 文本在 MiniCheck 14K 上微调，
不加载 GMN / 图注入。与 graph_match_nli 使用相同数据划分与相近超参，便于公平对比。

用法:
  python ablation/nli_text/train.py
  python ablation/nli_text/train.py --config configs/ablation.yaml
"""
import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import NLITextDataset
from graph_match_nli.nli_labels import (
    build_class_weights,
    dataset_labels_to_nli,
    nli_logits_to_support_preds,
    nli_logits_to_support_probs,
    resolve_nli_label_spec,
)
from utils.io_utils import load_yaml_config
from utils.path_utils import configure_dist_process_logging, log_rank0


def load_config(path: str) -> dict:
    return load_yaml_config(path)['ablation']['nli_ft']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'token_type_ids': torch.stack([b['token_type_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def evaluate_model(model, loader, device, criterion, label_spec, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)
            with autocast('cuda', enabled=use_amp):
                out = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    token_type_ids=batch['token_type_ids'].to(device),
                )
                logits = out.logits
                labels_nli = dataset_labels_to_nli(labels, label_spec)
                loss = criterion(logits, labels_nli)
            total_loss += loss.item() * labels.size(0)

            probs = nli_logits_to_support_probs(logits, label_spec)
            preds = nli_logits_to_support_preds(logits, label_spec)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    n = len(all_labels)
    return (
        total_loss / n,
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average='binary', zero_division=0),
        roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    )


def run_training(local_rank: int, world_size: int, config: dict, base_dir: str):
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    configure_dist_process_logging()
    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        dist.init_process_group(backend=backend, init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        _dev = config['model']['device']
        if _dev == 'cuda' and not torch.cuda.is_available():
            _dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        elif _dev == 'mps' and not torch.backends.mps.is_available():
            _dev = 'cpu'
        device = torch.device(_dev)

    is_main = (local_rank == 0)
    cfg_dir = os.path.join(base_dir, 'configs')

    def resolve(p):
        return os.path.normpath(os.path.join(cfg_dir, p))

    train_path = resolve(config['data']['train_parquet'])
    val_path = resolve(config['data']['val_parquet'])
    nli_model_path = resolve(config['model']['nli_model_path'])
    if not os.path.isdir(nli_model_path):
        raise FileNotFoundError(
            f"NLI 模型目录不存在: {nli_model_path}\n"
            f"请检查 configs/ablation.yaml 中 nli_ft.model.nli_model_path"
        )
    best_f1_path = resolve(config['training']['best_f1_path'])
    model_dir = os.path.dirname(best_f1_path)
    log_path = os.path.join(model_dir, 'train.log')

    log_file = None
    if is_main:
        os.makedirs(model_dir, exist_ok=True)
        log_file = open(log_path, 'a', encoding='utf-8')
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"NLI text-only FT started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'=' * 60}\n")
        log_file.flush()

    def log(msg):
        if not is_main:
            return
        print(msg)
        if log_file is not None:
            log_file.write(msg + '\n')
            log_file.flush()

    set_seed(config['training'].get('seed', 42) + local_rank)
    log(f"[NLI-Text-FT] device={device} | world_size={world_size}")
    log(f"  backbone: {nli_model_path}")
    log(f"  checkpoint: {best_f1_path}")

    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)
    train_ds = NLITextDataset(train_path, tokenizer, max_length=config['model']['max_length'])
    val_ds = NLITextDataset(val_path, tokenizer, max_length=config['model']['max_length'])

    batch_size = config['training']['batch_size']
    val_batch_size = config['training'].get('val_batch_size', batch_size)

    if is_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            drop_last=True, collate_fn=_collate,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate,
        )
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, collate_fn=_collate)

    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(device).float()
    label_spec = resolve_nli_label_spec(model.config.id2label)
    log(
        f"  label_spec: entailment={label_spec.entailment_id}, "
        f"hallucination={label_spec.hallucination_id}, num_labels={label_spec.num_labels}"
    )

    freeze_layers = int(config['model'].get('freeze_nli_layers', 0))
    if freeze_layers > 0 and hasattr(model, 'deberta'):
        for i, layer in enumerate(model.deberta.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        log(f"  冻结 DeBERTa 前 {freeze_layers} 层")

    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    encoder_params = []
    head_params = []
    raw_model = model.module if is_dist else model
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('classifier.') or name.startswith('pooler.'):
            head_params.append(p)
        else:
            encoder_params.append(p)

    lr_base = config['training']['learning_rate']
    lr_encoder = lr_base * config['training'].get('encoder_lr_ratio', 0.1)
    optimizer = AdamW([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': head_params, 'lr': lr_base},
    ], weight_decay=config['training'].get('weight_decay', 0.01))

    num_epochs = config['training']['num_epochs']
    accum_steps = config['training'].get('accum_steps', 1)
    steps_per_ep = (len(train_loader) + accum_steps - 1) // accum_steps
    total_steps = num_epochs * steps_per_ep
    warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    labels_arr = train_ds.df['label'].values
    class_weights = build_class_weights(labels_arr, label_spec, device)
    label_smoothing = float(config['training'].get('label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    patience = config['training']['patience']
    best_val_f1 = 0.0
    patience_cnt = 0

    log(f"\n开始训练 ({num_epochs} epochs, patience={patience})\n")
    for epoch in range(1, num_epochs + 1):
        if is_dist:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        total_loss, all_preds, all_labels = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}", disable=not is_main)
        for batch_idx, batch in enumerate(pbar):
            labels = batch['labels'].to(device)
            is_boundary = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            sync_ctx = model.no_sync() if (is_dist and not is_boundary) else nullcontext()

            with sync_ctx:
                with autocast('cuda', enabled=use_amp):
                    logits = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        token_type_ids=batch['token_type_ids'].to(device),
                    ).logits
                    labels_nli = dataset_labels_to_nli(labels, label_spec)
                    raw_loss = criterion(logits, labels_nli)
                    loss = raw_loss / accum_steps
                scaler.scale(loss).backward()

            if is_boundary:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += raw_loss.item() * labels.size(0)
            preds = nli_logits_to_support_preds(logits, label_spec).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            if is_main:
                pbar.set_postfix({'loss': f'{raw_loss.item():.4f}'})

        n_local = len(all_labels)
        if is_dist:
            loss_sum = torch.tensor([total_loss, float(n_local)], device=device)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            train_loss = loss_sum[0].item() / loss_sum[1].item()
            gathered_preds, gathered_labels = [None] * world_size, [None] * world_size
            dist.all_gather_object(gathered_preds, all_preds)
            dist.all_gather_object(gathered_labels, all_labels)
            all_preds = [p for sub in gathered_preds for p in sub]
            all_labels = [l for sub in gathered_labels for l in sub]
        else:
            train_loss = total_loss / n_local

        train_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        if is_main:
            val_loss, val_acc, val_f1, val_auc = evaluate_model(
                model, val_loader, device, criterion, label_spec, use_amp
            )
        else:
            val_loss = val_acc = val_f1 = val_auc = 0.0

        stop_flag = torch.zeros(1, device=device)
        if is_main:
            log(
                f"  Epoch {epoch:02d} | Train L={train_loss:.4f} F1={train_f1:.4f} | "
                f"Val L={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}"
            )
            model_to_save = model.module if is_dist else model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'val_loss': val_loss, 'val_f1': val_f1,
                    'val_acc': val_acc, 'val_auc': val_auc,
                    'config': config,
                    'label_spec': {
                        'num_labels': label_spec.num_labels,
                        'entailment_id': label_spec.entailment_id,
                        'hallucination_id': label_spec.hallucination_id,
                    },
                }, best_f1_path)
                log(f"  📈 best_f1 -> {best_f1_path} (Val F1={best_val_f1:.4f})")
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log(f"\n⛔ Early stop. Best Val F1={best_val_f1:.4f}")
                    stop_flag[0] = 1.0

        if is_dist:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1.0:
            break

    if is_main:
        log(f"\n完成。Best Val F1={best_val_f1:.4f}")
        if log_file is not None:
            log_file.close()
    if is_dist:
        dist.destroy_process_group()


def main_worker(local_rank, world_size, config_path, base_dir):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12357')
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    run_training(local_rank, world_size, load_config(config_path), base_dir)


def main():
    parser = argparse.ArgumentParser(description="NLI 纯文本微调（消融 w/o graph）")
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    base_dir = _PROJ_ROOT
    config_path = args.config or os.path.join(base_dir, 'configs', 'ablation.yaml')
    config = load_config(config_path)
    tp = config['training'].get('tensor_parallel_size', 1)
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if tp > 1 and not is_torchrun:
        import torch.multiprocessing as mp
        log_rank0(f"[Init] DDP nprocs={tp}")
        mp.spawn(main_worker, nprocs=tp, args=(tp, config_path, base_dir))
    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        run_training(local_rank, world_size, config, base_dir)


if __name__ == '__main__':
    main()
