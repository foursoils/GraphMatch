"""
NLI-Graph 融合模型训练脚本

流程:
  1. 加载 configs/graph_match_nli.yaml
  2. 构建 NLIGraphDataset（tokenizer 输入 + 图对）
  3. 训练 NLIGraphClassifier（GMN + DeBERTa 中间层注入）
  4. Early Stopping，按 val F1 保存最优检查点
  5. 训练 log 写入模型目录下的 train.log

多卡:
  配置 training.tensor_parallel_size > 1 时，自动通过 mp.spawn 拉起多进程 DDP 训练
  （每卡一个进程，梯度通过 NCCL all-reduce 同步）。也兼容 torchrun 启动方式。
"""
import os
import sys
import argparse
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')  # MPS 不限制上限，避免 OOM
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import NLIGraphClassifier
from dataset import NLIGraphDataset
from nli_labels import (
    build_class_weights,
    dataset_labels_to_nli,
    nli_logits_to_support_preds,
    nli_logits_to_support_probs,
)
from utils.path_utils import configure_dist_process_logging, log_rank0
from utils.io_utils import load_yaml_config


def load_config(path: str) -> dict:
    return load_yaml_config(path)['nli_graph']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion, label_spec, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            input_ids      = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            labels = batch.y.view(-1).long()

            with autocast('cuda', enabled=use_amp):
                logits = model(input_ids, attention_mask, token_type_ids, batch)
                labels_nli = dataset_labels_to_nli(labels, label_spec)
                loss = criterion(logits, labels_nli)
            total_loss += loss.item() * labels.size(0)

            probs = nli_logits_to_support_probs(logits, label_spec)
            preds = nli_logits_to_support_preds(logits, label_spec)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, acc, f1, auc


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程（单卡 / DDP 多卡通用，由 local_rank / world_size 决定行为）
# ─────────────────────────────────────────────────────────────────────────────
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

    train_path     = resolve(config['data']['train_parquet'])
    val_path       = resolve(config['data']['val_parquet'])
    nli_model_path = resolve(config['model']['nli_model_path'])
    emb_model_path = resolve(config['model']['embedding_model_path'])
    best_f1_path   = resolve(config['training']['best_f1_path'])
    model_dir      = os.path.dirname(best_f1_path)
    log_path       = os.path.join(model_dir, 'train.log')

    log_file = None
    if is_main:
        os.makedirs(model_dir, exist_ok=True)
        log_file = open(log_path, 'a', encoding='utf-8')
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"Training started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    log(f"使用设备: {device} | world_size={world_size}")
    log(f"模型保存路径: {best_f1_path}")
    log(f"训练日志: {log_path}")

    tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False)
    train_ds = NLIGraphDataset(
        train_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device),
        embed_cache_path=resolve(config['data']['train_embed_file']) if config['data'].get('train_embed_file') else None,
    )
    val_ds = NLIGraphDataset(
        val_path, tokenizer, emb_model_path,
        max_length=config['model']['max_length'],
        device=str(device),
        embed_cache_path=resolve(config['data']['val_embed_file']) if config['data'].get('val_embed_file') else None,
    )
    batch_size = config['training']['batch_size']

    if is_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True,
            follow_batch=['x_s', 'x_t']
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            follow_batch=['x_s', 'x_t']
        )
    # 验证集不做分布式切分，只在主进程上完整评估，避免多进程重复计算 / 写文件冲突
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        follow_batch=['x_s', 'x_t']
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    log("[4/5] 初始化 NLI-Graph 融合模型...")
    model = NLIGraphClassifier(
        nli_model_path      = nli_model_path,
        node_input_dim      = config['model']['node_input_dim'],
        edge_input_dim      = config['model']['node_input_dim'],
        node_hidden_dim     = config['model']['node_hidden_dim'],
        num_prop_layers     = config['model']['num_prop_layers'],
        inject_layer_k      = config['model']['inject_layer_k'],
        num_heads           = config['model']['num_heads'],
        dropout             = config['model']['dropout'],
        freeze_nli_layers   = config['model'].get('freeze_nli_layers', 0),
        num_labels          = config['model'].get('num_labels', 2),
    ).to(device).float()   # MPS 上强制 float32，避免 f16/f32 混合导致崩溃

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label_spec = model.label_spec
    log(f"  总参数: {total_params:,} | 可训练: {trainable_params:,}")
    log(
        f"  NLI 标签映射: entailment={label_spec.entailment_id}, "
        f"hallucination={label_spec.hallucination_id}, num_labels={label_spec.num_labels}"
    )

    # ── 差异化学习率：DeBERTa 用更小 lr，GMN + 注入层用较大 lr ────────────
    # 注意：必须在 DDP 包装之前拿到参数引用（DDP 不会转发 .nli_encoder 等属性访问）
    deberta_params = [p for p in model.nli_encoder.parameters() if p.requires_grad]
    other_params   = [p for p in model.parameters()
                      if p.requires_grad and not any(p is q for q in deberta_params)]

    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    lr_base   = config['training']['learning_rate']
    lr_deberta = lr_base * config['training'].get('deberta_lr_ratio', 0.1)

    optimizer = AdamW([
        {'params': deberta_params, 'lr': lr_deberta},
        {'params': other_params,   'lr': lr_base},
    ], weight_decay=config['training'].get('weight_decay', 0.01))

    # 学习率：Linear Warmup → Cosine Decay（到末尾自然降到 ~0，掐死后期过拟合空间）
    num_epochs    = config['training']['num_epochs']
    accum_steps   = config['training'].get('accum_steps', 1)
    steps_per_ep  = (len(train_loader) + accum_steps - 1) // accum_steps
    total_steps   = num_epochs * steps_per_ep
    warmup_steps  = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    log(f"  调度器: accum_steps={accum_steps} | warmup={warmup_steps} steps / total={total_steps} steps (cosine decay)")
    if is_dist:
        log(f"  多卡: world_size={world_size}，全局等效 batch = world_size × batch_size × accum_steps = "
            f"{world_size * batch_size * accum_steps}")

    # 混合精度
    use_amp = (device.type == 'cuda')
    scaler  = GradScaler('cuda', enabled=use_amp)

    # 类别权重（在 NLI 类别空间计算，与标签映射保持一致）
    labels_arr = train_ds.df['label'].values
    pos = int(np.sum(labels_arr == 1))
    neg = int(np.sum(labels_arr == 0))
    class_weights = build_class_weights(labels_arr, label_spec, device)
    log(f"  数据分布: 支持(1)={pos}, 幻觉(0)={neg} | NLI类别权重={class_weights.tolist()}")

    label_smoothing = float(config['training'].get('label_smoothing', 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    log(f"  Label smoothing: {label_smoothing}")

    patience        = config['training']['patience']
    log(f"  监控指标: val_f1（升高更好）")

    best_val_f1     = 0.0
    patience_cnt    = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────
    log(f"\n[5/5] 开始训练（共 {num_epochs} epoch，早停耐心={patience}）\n")
    for epoch in range(1, num_epochs + 1):
        if is_dist:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        total_loss, all_preds, all_labels = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}", unit="batch", disable=not is_main)
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)

            input_ids      = batch.input_ids
            attention_mask = batch.attention_mask
            token_type_ids = batch.token_type_ids
            labels         = batch.y.view(-1).long()

            is_accum_boundary = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            # 梯度累积期间跳过 DDP 的 all-reduce，只在累积边界同步一次，减少通信开销
            sync_ctx = model.no_sync() if (is_dist and not is_accum_boundary) else nullcontext()

            with sync_ctx:
                with autocast('cuda', enabled=use_amp):
                    logits = model(input_ids, attention_mask, token_type_ids, batch)
                    labels_nli = dataset_labels_to_nli(labels, label_spec)
                    raw_loss = criterion(logits, labels_nli)
                    loss   = raw_loss / accum_steps

                scaler.scale(loss).backward()

            if is_accum_boundary:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()   # cosine 调度按 step 推进

            total_loss += raw_loss.item() * labels.size(0)
            preds = nli_logits_to_support_preds(logits, label_spec).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            if is_main:
                running_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'loss': f'{raw_loss.item():.4f}',
                    'acc':  f'{running_acc:.4f}',
                })

        # 汇总各 rank 的训练损失 / 预测，得到全局训练指标
        n_local = len(all_labels)
        if is_dist:
            loss_sum_tensor = torch.tensor([total_loss, float(n_local)], device=device)
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_sum_tensor[0].item() / loss_sum_tensor[1].item()

            gathered_preds  = [None] * world_size
            gathered_labels = [None] * world_size
            dist.all_gather_object(gathered_preds, all_preds)
            dist.all_gather_object(gathered_labels, all_labels)
            all_preds_g  = [p for sub in gathered_preds for p in sub]
            all_labels_g = [l for sub in gathered_labels for l in sub]
        else:
            train_loss = total_loss / n_local
            all_preds_g, all_labels_g = all_preds, all_labels

        train_acc  = accuracy_score(all_labels_g, all_preds_g)
        train_f1   = f1_score(all_labels_g, all_preds_g, average='binary', zero_division=0)

        # 验证只在主进程上跑（val_loader 未做分布式切分，覆盖完整验证集）
        if is_main:
            val_loss, val_acc, val_f1, val_auc = evaluate(
                model, val_loader, device, criterion, label_spec, use_amp
            )
        else:
            val_loss = val_acc = val_f1 = val_auc = 0.0

        if is_main:
            cur_lr_main    = optimizer.param_groups[1]['lr']
            cur_lr_deberta = optimizer.param_groups[0]['lr']

            # 过拟合警告：Train-Val Loss 差距 + Val F1 差距
            gap_loss = val_loss - train_loss
            gap_f1   = train_f1 - val_f1
            warn = ""
            if gap_loss > 0.4 or gap_f1 > 0.10:
                warn = "  ⚠️ 过拟合迹象"

            log(
                f"  Epoch {epoch:02d} | "
                f"Train L={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
                f"Val L={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f} | "
                f"LR(main/deberta)={cur_lr_main:.2e}/{cur_lr_deberta:.2e}{warn}"
            )

        # ── 早停判断与 checkpoint 保存：只在主进程决策，再广播给其余 rank ──
        stop_flag = torch.zeros(1, device=device)
        if is_main:
            improved_f1   = val_f1 > best_val_f1
            model_to_save = model.module if is_dist else model

            if improved_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'val_loss': val_loss, 'val_f1': val_f1,
                    'val_acc':  val_acc,  'val_auc': val_auc,
                    'config':   config,
                }, best_f1_path)
                log(f"  📈 best_f1 已更新 (Val F1={best_val_f1:.4f}) -> {best_f1_path}")
                patience_cnt = 0
            else:
                patience_cnt += 1
                log(f"  ⏳ 早停计数: {patience_cnt}/{patience} (基于 val_f1)")
                if patience_cnt >= patience:
                    log(f"\n⛔ 早停触发！最佳 Val F1={best_val_f1:.4f}")
                    stop_flag[0] = 1.0

        if is_dist:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1.0:
            break

    if is_main:
        log(f"\n训练完成！最佳 Val F1={best_val_f1:.4f}")
        log(f"  最优检查点: {best_f1_path}")
        log(f"  训练日志: {log_path}")
        if log_file is not None:
            log_file.write(f"Training finished {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.close()

    if is_dist:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 多卡启动入口
# ─────────────────────────────────────────────────────────────────────────────
def main_worker(local_rank: int, world_size: int, config_path: str, base_dir: str):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12356')
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    config = load_config(config_path)
    run_training(local_rank, world_size, config, base_dir)


def main():
    parser = argparse.ArgumentParser(description="NLI-Graph 融合模型训练")
    parser.add_argument('--config', type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(base_dir, 'configs', 'graph_match_nli.yaml')
    config      = load_config(config_path)

    tensor_parallel_size = config['training'].get('tensor_parallel_size', 1)
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if tensor_parallel_size > 1 and not is_torchrun:
        import torch.multiprocessing as mp
        log_rank0(f"[Init] 检测到配置 tensor_parallel_size={tensor_parallel_size}，正在自动拉起多卡 DDP 分布式训练...")
        mp.spawn(main_worker, nprocs=tensor_parallel_size, args=(tensor_parallel_size, config_path, base_dir))
    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        run_training(local_rank, world_size, config, base_dir)


if __name__ == '__main__':
    main()
