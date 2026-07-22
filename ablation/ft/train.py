"""ablation/ft LoRA 纯文本微调入口。"""

import os
import sys
from datetime import datetime

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)

from ablation.ft.common import maybe_relaunch_multigpu

maybe_relaunch_multigpu('training')

import argparse
import json

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ablation.ft.common import compute_bacc, load_config, parse_binary_pred, seed_everything
from ablation.ft.dataset import HalluTextDataset, text_collate_fn
from ablation.ft.model import LoraHalluModel
from utils.path_utils import configure_dist_process_logging, resolve_num_workers, resolve_path


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.streams[0], 'isatty', lambda: False)()


def setup_train_log(log_dir: str, config_path: str, config: dict):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_{ts}.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    orig_stdout = sys.stdout
    sys.stdout = _TeeWriter(orig_stdout, log_file)
    print(f"[Log] 训练日志: {log_path}")
    print(f"[Log] 配置文件: {config_path}")
    print(f"[Log] 启动时间: {datetime.now().isoformat(timespec='seconds')}")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    return log_path, log_file, orig_stdout


def teardown_train_log(log_file, orig_stdout):
    if log_file is not None:
        sys.stdout = orig_stdout
        log_file.close()


@torch.no_grad()
def validate(model, loader, device, accelerator):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc='  [Val]', leave=False, disable=not accelerator.is_main_process):
        result = accelerator.unwrap_model(model).inference(batch)
        pred_tensor = torch.tensor(
            [parse_binary_pred(p) for p in result['pred']], dtype=torch.long, device=device
        )
        label_tensor = torch.tensor([int(l) for l in result['label']], dtype=torch.long, device=device)
        all_preds.extend(accelerator.gather_for_metrics(pred_tensor).cpu().tolist())
        all_labels.extend(accelerator.gather_for_metrics(label_tensor).cpu().tolist())

    valid = [(p, l) for p, l in zip(all_preds, all_labels) if p != -1]
    bacc = compute_bacc([p for p, _ in valid], [l for _, l in valid]) if valid else 0.0
    model.train()
    return bacc, len(valid) / max(len(all_preds), 1)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ablation.yaml')
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    data_cfg, model_cfg, train_cfg = config['data'], config['model'], config['training']

    gpu_ids = train_cfg.get('gpu_ids')
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    seed_everything(train_cfg.get('seed', 42))

    train_file = resolve_path(data_cfg['train_file'])
    val_file = resolve_path(data_cfg['val_file'])
    output_dir = resolve_path(train_cfg['output_dir'])
    sys_prompt = model_cfg.get('system_prompt_path', 'prompts/hallu_detect/system_prompt.txt')
    user_prompt = model_cfg.get('user_prompt_path', 'prompts/hallu_detect/user_prompt.txt')

    grad_accum = train_cfg.get('grad_accum_steps', 16)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=train_cfg.get('mixed_precision', 'bf16'),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    log_file, orig_stdout = None, sys.stdout
    if accelerator.is_main_process:
        _, log_file, orig_stdout = setup_train_log(
            resolve_path(train_cfg.get('log_dir', 'log/ablation_ft')),
            config_path,
            config,
        )

    try:
        configure_dist_process_logging()
        accelerator.print('\n[1/4] 初始化模型...')
        model = LoraHalluModel(config, device=accelerator.device, apply_lora=True)
        model.print_trainable_params()

        accelerator.print('\n[2/4] 构建数据集...')
        train_ds = HalluTextDataset(train_file, sys_prompt, user_prompt, is_train=True)
        val_ds = HalluTextDataset(val_file, sys_prompt, user_prompt, is_train=False)

        if train_cfg.get('val_sample', False):
            total_val = len(val_ds)
            n = min(train_cfg.get('val_sample_size', 300), total_val)
            val_ds = Subset(val_ds, np.random.default_rng(42).choice(total_val, n, replace=False).tolist())
            accelerator.print(f'[Dataset] 验证集采样: {n}/{total_val}')

        workers = resolve_num_workers(train_cfg.get('num_workers', 2))
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg.get('batch_size', 1), shuffle=True,
            num_workers=workers, collate_fn=text_collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg.get('eval_batch_size', 4), shuffle=False,
            num_workers=workers, collate_fn=text_collate_fn, pin_memory=True,
        )

        accelerator.print('\n[3/4] 优化器与调度器...')
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=train_cfg.get('lr', 5e-6),
            weight_decay=train_cfg.get('weight_decay', 0.0),
        )

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        num_epochs = train_cfg.get('num_epochs', 1)
        total_steps = max(1, (len(train_loader) // grad_accum) * num_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * train_cfg.get('warmup_ratio', 0.1)),
            num_training_steps=total_steps,
        )
        scheduler = accelerator.prepare(scheduler)

        accelerator.print(f'\n[4/4] 开始训练（{num_epochs} epoch）\n')
        patience = train_cfg.get('patience', 3)
        best_bacc, no_improve = -1.0, 0
        best_ckpt = os.path.join(output_dir, 'best_model.pt')
        history = []

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader, desc=f'Epoch {epoch:02d}/{num_epochs}',
                dynamic_ncols=True, disable=not accelerator.is_main_process,
            )
            for step, batch in enumerate(pbar, 1):
                with accelerator.accumulate(model):
                    loss = model(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.num_processes > 1:
                    loss_val = accelerator.reduce(loss.detach(), 'mean').item()
                else:
                    loss_val = loss.item()
                epoch_loss += loss_val
                if accelerator.is_main_process:
                    pbar.set_postfix(loss=f'{epoch_loss / step:.4f}')

            n = len(train_loader)
            accelerator.print(f'\nEpoch {epoch:02d} | train_loss={epoch_loss / n:.4f}')

            bacc, parse_rate = validate(model, val_loader, accelerator.device, accelerator)
            accelerator.print(f'         | val_BAcc={bacc:.4f}  parse_rate={parse_rate:.2%}')

            if accelerator.is_main_process:
                history.append({'epoch': epoch, 'train_loss': epoch_loss / n, 'val_bacc': bacc})
                if bacc > best_bacc:
                    best_bacc, no_improve = bacc, 0
                    m = accelerator.unwrap_model(model)
                    try:
                        m.llm.save_pretrained(os.path.join(output_dir, 'lora_adapter'))
                    except Exception as e:
                        print(f'  [Warn] LoRA 保存失败: {e}')
                    accelerator.save(
                        {'epoch': epoch, 'val_bacc': bacc, 'train_loss': epoch_loss / n},
                        best_ckpt,
                    )
                    print(f'  最优模型已保存 (BAcc={bacc:.4f})')
                else:
                    no_improve += 1
                    print(f'  无改善 ({no_improve}/{patience})')

            if accelerator.num_processes > 1:
                import torch.distributed as dist
                t = torch.tensor(no_improve, dtype=torch.long, device=accelerator.device)
                dist.broadcast(t, src=0)
                no_improve = t.item()

            if no_improve >= patience:
                accelerator.print('  Early Stopping 触发，训练结束。')
                break

        if accelerator.is_main_process:
            with open(os.path.join(output_dir, 'train_history.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            print(f'\n训练完成！最优 val_BAcc={best_bacc:.4f}\n检查点: {best_ckpt}')
    finally:
        if accelerator.is_main_process:
            teardown_train_log(log_file, orig_stdout)


if __name__ == '__main__':
    train()
