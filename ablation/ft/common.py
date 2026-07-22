"""ablation/ft 共享工具。"""

import os
import random
import re
import subprocess
import sys

import numpy as np
import torch
import yaml

from utils.path_utils import get_project_root, resolve_path

CONFIG_KEY = 'ft'
CONFIG_DEFAULT = 'configs/ablation.yaml'


def load_config(path: str) -> dict:
    config_path = path if os.path.isabs(path) else resolve_path(path)
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    ablation = raw.get('ablation') or {}
    if CONFIG_KEY not in ablation:
        raise KeyError(f"配置缺少 ablation.{CONFIG_KEY}")
    return ablation[CONFIG_KEY]


def parse_binary_pred(text: str) -> int:
    """从生成文本解析 0/1；失败返回 -1。"""
    if not text:
        return -1
    cleaned = text.strip().lower()
    if cleaned in ('1', 'yes'):
        return 1
    if cleaned in ('0', 'no'):
        return 0

    for pattern in (
        r'answer\s+is\s*:\s*(yes|no)',
        r'answer\s+is\s*:\s*(1|0)',
    ):
        m = re.search(pattern, cleaned)
        if m:
            return 1 if m.group(1) in ('yes', '1') else 0

    matches = re.findall(r'\b(yes|no|1|0)\b', cleaned)
    if matches:
        return 1 if matches[-1] in ('yes', '1') else 0
    return -1


def compute_bacc(preds: list, labels: list) -> float:
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        if p == 1 and l == 1:
            tp += 1
        elif p == 1 and l == 0:
            fp += 1
        elif p == 0 and l == 0:
            tn += 1
        elif p == 0 and l == 1:
            fn += 1
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    return (recall_pos + recall_neg) / 2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompt(path: str, fallback: str) -> str:
    full = path if os.path.isabs(path) else resolve_path(path)
    if os.path.exists(full):
        with open(full, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return fallback


def maybe_relaunch_multigpu(section: str, config_default: str = CONFIG_DEFAULT):
    """多卡时在导入 torch 前自动拉起 accelerate launch。"""
    if any(k in os.environ for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')):
        return

    proj_root = get_project_root()
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_default)
    args, _ = parser.parse_known_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(proj_root, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    config = (raw.get('ablation') or {}).get(CONFIG_KEY) or {}
    section_cfg = config.get(section, {})
    train_cfg = config.get('training', {})

    gpu_ids = section_cfg.get('gpu_ids')
    if gpu_ids is None or gpu_ids == '':
        return

    gpus = [x.strip() for x in str(gpu_ids).split(',') if x.strip()]
    if len(gpus) <= 1:
        return

    print(f"\n[Self-Launcher] 多卡 ({gpu_ids})，通过 accelerate launch 启动...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    cmd = [
        sys.executable, '-m', 'accelerate.commands.launch',
        f'--num_processes={len(gpus)}',
        '--num_machines=1',
        f'--mixed_precision={train_cfg.get("mixed_precision", "bf16")}',
        '--dynamo_backend=no',
        sys.argv[0],
        *sys.argv[1:],
    ]
    sys.exit(subprocess.run(cmd).returncode)
