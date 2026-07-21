"""
GraphCheck 消融实验 - 训练程序
================================
功能：
  - 加载 ablation.yaml 中 check 节点的配置
  - 构建 KGDataset + DataLoader
  - 初始化 GraphCheck 模型（冻结 LLM，只训练 GNN + Projector）
  - 执行训练 + 验证循环，支持 Early Stopping
  - 保存最优检查点，并在测试集上评估 Balanced Accuracy

用法：
  python check_train.py
  python check_train.py --config ../../configs/ablation.yaml
"""

import os
import sys
import gc
import json
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from model.graphcheck import GraphCheck

# Add project root to sys.path so we can reuse the shared utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_utils import resolve_path as _resolve_path_from_root
from utils.io_utils import load_yaml_config


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """加载 YAML 文件，返回 ablation.check 节点。"""
    return load_yaml_config(config_path)['ablation']['check']


def resolve_path(base_dir: str, rel_or_abs: str) -> str:
    """将相对路径解析为绝对路径（base_dir 已固定为项目根目录，故直接复用共享实现）。"""
    return _resolve_path_from_root(rel_or_abs)


def seed_everything(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(param_group, base_lr: float, cur_step: float, config: dict):
    """余弦 Warmup 学习率调度。"""
    warmup_epochs = config['training'].get('warmup_epochs', 1)
    num_epochs    = config['training']['num_epochs']
    if cur_step < warmup_epochs:
        lr = base_lr * (cur_step / warmup_epochs)
    else:
        progress = (cur_step - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    param_group['lr'] = lr


def get_balanced_accuracy(result_path: str) -> float:
    """从推理输出 jsonl 文件计算 Balanced Accuracy (BAcc)。"""
    tp = fp = tn = fn = 0
    with open(result_path, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            pred  = str(row['pred']).strip().lower()
            label = str(row['label']).strip().lower()
            # 将 "yes"/"true"/"1" 视为正例
            is_pos_pred  = pred  in ('yes', 'true', '1')
            is_pos_label = label in ('yes', 'true', '1')
            if is_pos_pred and is_pos_label:   tp += 1
            elif is_pos_pred:                  fp += 1
            elif is_pos_label:                 fn += 1
            else:                              tn += 1
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sens + spec) / 2.0


def save_checkpoint(model, optimizer, epoch: int, output_dir: str, is_best: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    ge = model.graph_encoder.module if hasattr(model.graph_encoder, 'module') else model.graph_encoder
    proj = model.projector.module if hasattr(model.projector, 'module') else model.projector
    ckpt = {
        'epoch': epoch,
        'gnn_state_dict': ge.state_dict(),
        'proj_state_dict': proj.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    path = os.path.join(output_dir, 'best_model.pt' if is_best else f'ckpt_epoch{epoch}.pt')
    torch.save(ckpt, path)
    print(f"  [Ckpt] 保存检查点: {path}")


def load_best_model(model, output_dir: str):
    path = os.path.join(output_dir, 'best_model.pt')
    ckpt = torch.load(path, map_location='cpu')
    ge = model.graph_encoder.module if hasattr(model.graph_encoder, 'module') else model.graph_encoder
    proj = model.projector.module if hasattr(model.projector, 'module') else model.projector
    ge.load_state_dict(ckpt['gnn_state_dict'])
    proj.load_state_dict(ckpt['proj_state_dict'])
    print(f"  [Ckpt] 加载最优模型: {path} (Epoch {ckpt['epoch']})")
    return model


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_training(local_rank, world_size, config):
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    # ---- 检查是否开启 DDP ----
    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        dist.init_process_group(backend=backend, init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 随机种子 ----
    seed = config['training'].get('seed', 42)
    seed_everything(seed + local_rank)

    # ---- 输出目录 ----
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = resolve_path(base_dir, config['training']['output_dir'])
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Init] 检查点/结果输出目录: {output_dir}")

    # ---- 数据集 ----
    from dataset import GraphCheckDataset, graphcheck_collate_fn

    train_path = resolve_path(base_dir, config['data']['train_file'])
    val_path   = resolve_path(base_dir, config['data']['val_file'])

    embed_path = resolve_path(base_dir, config['model']['embed_model_path'])
    
    # 构建 Dataset (注意: 在线 Embedding 比较慢，且 DataLoader 不能使用多个 worker 以免显存冲突)
    train_dataset = GraphCheckDataset(train_path, embed_path)
    val_dataset   = GraphCheckDataset(val_path,   embed_path)

    batch_size      = config['training']['batch_size']
    eval_batch_size = config['training'].get('eval_batch_size', batch_size)
    num_workers     = config['training'].get('num_workers', 4)

    # 动态根据是否使用预计算特征决定 num_workers
    train_num_workers = num_workers if train_dataset.use_precomputed else 0
    val_num_workers   = num_workers if val_dataset.use_precomputed else 0
    
    if local_rank == 0:
        print(f"[Init] DataLoader num_workers -> train: {train_num_workers}, val/test: {val_num_workers}")

    if is_dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=train_sampler, drop_last=True, pin_memory=True,
                                  num_workers=train_num_workers, collate_fn=graphcheck_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,  drop_last=True,  pin_memory=True,
                                  num_workers=train_num_workers, collate_fn=graphcheck_collate_fn)

    val_loader   = DataLoader(val_dataset,   batch_size=eval_batch_size,
                              shuffle=False, drop_last=False, pin_memory=True,
                              num_workers=val_num_workers, collate_fn=graphcheck_collate_fn)
    # 因为原配置去掉了 test_dataset 设置，若需要在测试集评估，您可以自行加测，这里暂用 val_dataset 演示评估
    test_loader  = DataLoader(val_dataset,  batch_size=eval_batch_size,
                              shuffle=False, drop_last=False, pin_memory=True,
                              num_workers=val_num_workers, collate_fn=graphcheck_collate_fn)

    # ---- 模型 ----
    if local_rank == 0:
        print("[Init] 初始化 GraphCheck 模型 ...")
    model = GraphCheck(config)
    if local_rank == 0:
        model.print_trainable_params()

    # 包装分布式
    if is_dist:
        model.graph_encoder = DDP(model.graph_encoder, device_ids=[local_rank], output_device=local_rank)
        model.projector = DDP(model.projector, device_ids=[local_rank], output_device=local_rank)

    # ---- 优化器（只优化可训练参数）----
    lr         = config['training'].get('lr',           1e-4)
    weight_decay = config['training'].get('weight_decay', 0.05)
    grad_steps = config['training'].get('grad_steps',   4)
    patience   = config['training'].get('patience',     3)
    num_epochs = config['training']['num_epochs']

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': trainable_params, 'lr': lr, 'weight_decay': weight_decay}],
        betas=(0.9, 0.95),
    )

    # ---- 训练循环 ----
    num_steps   = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_steps), desc="Training") if local_rank == 0 else None
    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(num_epochs):
        if is_dist:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            clip_grad_norm_(trainable_params, max_norm=0.1)

            if (step + 1) % grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch, config)

            optimizer.step()
            epoch_loss += loss.item()
            if progress_bar:
                progress_bar.update(1)

        avg_train_loss = epoch_loss / len(train_loader)
        
        # 分布式汇总 loss
        if is_dist:
            loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = loss_tensor.item() / dist.get_world_size()

        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        
        # 仅在 rank 0 验证
        if not is_dist or local_rank == 0:
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += model(batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

        # 同步早停信号
        early_stop = torch.tensor(0, device=device)
        if not is_dist or local_rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                save_checkpoint(model, optimizer, epoch, output_dir, is_best=True)

            if epoch - best_epoch >= patience:
                print(f"[EarlyStopping] 连续 {patience} 轮无改善，提前停止。")
                early_stop = torch.tensor(1, device=device)

        if is_dist:
            dist.broadcast(early_stop, src=0)

        if early_stop.item() == 1:
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # ---- 测试集评估 ----
    if not is_dist or local_rank == 0:
        result_path = os.path.join(output_dir, 'test_results.jsonl')
        print(f"\n[Eval] 在测试集上推理，结果保存至: {result_path}")

        # 解包 DDP 模块
        ge_unwrapped = model.graph_encoder.module if hasattr(model.graph_encoder, 'module') else model.graph_encoder
        proj_unwrapped = model.projector.module if hasattr(model.projector, 'module') else model.projector

        orig_ge, orig_proj = model.graph_encoder, model.projector
        model.graph_encoder, model.projector = ge_unwrapped, proj_unwrapped

        model = load_best_model(model, output_dir)
        model.eval()

        with open(result_path, 'w') as f:
            for batch in tqdm(test_loader, desc="Inferring"):
                with torch.no_grad():
                    output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + '\n')

        bacc = get_balanced_accuracy(result_path)
        print(f"[Result] Test Balanced Accuracy: {bacc:.4f}")

        # 还原回来
        model.graph_encoder, model.projector = orig_ge, orig_proj

    if is_dist:
        dist.destroy_process_group()

    torch.cuda.empty_cache()
    gc.collect()


def main_worker(local_rank, world_size, config_path):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    # 重新加载 config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(config_path)
    run_training(local_rank, world_size, config)


def main():
    parser = argparse.ArgumentParser(description="GraphCheck Ablation: Train")
    parser.add_argument('--config', type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    config_path = args.config or os.path.join(base_dir, 'configs', 'ablation.yaml')
    config      = load_config(config_path)

    tensor_parallel_size = config['training'].get('tensor_parallel_size', 1)
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if tensor_parallel_size > 1 and not is_torchrun:
        import torch.multiprocessing as mp
        print(f"[Init] 检测到配置 tensor_parallel_size={tensor_parallel_size}，正在自动拉起多卡 DDP 分布式训练...")
        mp.spawn(main_worker, nprocs=tensor_parallel_size, args=(tensor_parallel_size, config_path))
    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        run_training(local_rank, world_size, config)


if __name__ == '__main__':
    main()
