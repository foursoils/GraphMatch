"""
Token Count Analysis for Claim and Doc Fields
使用 Qwen3-Embedding-0.6B 的 tokenizer 统计各数据集中 claim 和 doc 的 token 数分布，
并统计负标签（label == 0）比例。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def resolve_device() -> str:
    """按 CUDA → MPS → CPU 选择可用设备（tokenize 本身不使用 GPU）。"""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] 检测到 CUDA: {name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] 检测到 MPS（Apple Silicon）")
        return "mps"
    print("[INFO] 未检测到 CUDA/MPS，使用 CPU")
    return "cpu"


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """加载 tokenizer（CPU 操作）。"""
    print(f"[INFO] 正在加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[INFO] Tokenizer 加载完成，vocab size = {tokenizer.vocab_size}")
    return tokenizer


def count_tokens(texts: list[str], tokenizer: AutoTokenizer, batch_size: int = 512) -> list[int]:
    """批量统计 token 数（纯 tokenizer 操作，在 CPU 上运行）。"""
    counts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  统计 token", leave=False):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


def analyze_labels(df: pd.DataFrame, label_col: str) -> dict | None:
    """统计负标签（label == 0）比例。"""
    if label_col not in df.columns:
        print(f"  [WARN] 列 '{label_col}' 不存在，跳过标签统计。可用列: {list(df.columns)}")
        return None

    labels = df[label_col]
    n = len(labels)
    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    n_other = n - n_neg - n_pos
    return {
        "col_key": label_col,
        "n_neg": n_neg,
        "n_pos": n_pos,
        "n_other": n_other,
        "neg_ratio": float(n_neg / n) if n else 0.0,
        "pos_ratio": float(n_pos / n) if n else 0.0,
    }


def _token_stats(arr: np.ndarray, col_key: str) -> dict:
    return {
        "col_key": col_key,
        "mean": float(np.mean(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "<512": int(np.sum(arr < 512)),
        ">512": int(np.sum(arr > 512)),
        ">1024": int(np.sum(arr > 1024)),
        ">2048": int(np.sum(arr > 2048)),
    }


def analyze_dataset(
    parquet_path: Path,
    tokenizer: AutoTokenizer,
    claim_col: str,
    doc_col: str,
    label_col: str,
    batch_size: int = 512,
) -> dict:
    """读取单个数据集并统计 token 分布与标签比例。"""
    df = pd.read_parquet(parquet_path)

    result = {
        "n_samples": len(df),
        "columns": list(df.columns),
        "label": analyze_labels(df, label_col),
    }

    token_counts_dict = {}
    for col_name, col_key in [("claim", claim_col), ("doc", doc_col)]:
        if col_key not in df.columns:
            print(f"  [WARN] 列 '{col_key}' 不存在，跳过。可用列: {list(df.columns)}")
            result[col_name] = None
            continue

        texts = df[col_key].fillna("").astype(str).tolist()
        token_counts = count_tokens(texts, tokenizer, batch_size=batch_size)
        token_counts_dict[col_name] = token_counts
        result[col_name] = _token_stats(np.array(token_counts), col_key)

    if "claim" in token_counts_dict and "doc" in token_counts_dict:
        combined = np.array(token_counts_dict["claim"]) + np.array(token_counts_dict["doc"])
        result["claim+doc"] = _token_stats(combined, f"{claim_col} + {doc_col}")

    return result


def print_report(dataset_name: str, stats: dict) -> None:
    """格式化打印单个数据集的统计报告。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  数据集: {dataset_name}")
    print(f"  样本数: {stats['n_samples']}")
    print(f"  数据列: {stats['columns']}")
    print(sep)

    label = stats.get("label")
    if label is None:
        print("\n  [LABEL] 列不存在，已跳过")
    else:
        n = stats["n_samples"]
        print(f"\n  [LABEL]  (列: '{label['col_key']}')")
        print(f"    负标签 (0): {label['n_neg']}  ({label['neg_ratio']*100:.1f}%)")
        print(f"    正标签 (1): {label['n_pos']}  ({label['pos_ratio']*100:.1f}%)")
        if label["n_other"]:
            print(f"    其他值    : {label['n_other']}  ({label['n_other']/n*100:.1f}%)")

    for field in ["claim", "doc", "claim+doc"]:
        s = stats.get(field)
        if s is None:
            print(f"  [{field.upper()}] 列不存在，已跳过")
            continue
        n = stats["n_samples"]
        print(f"\n  [{field.upper()}]  (列: '{s['col_key']}')")
        print(f"    均值    : {s['mean']:.1f} tokens")
        print(f"    最小值  : {s['min']} tokens")
        print(f"    最大值  : {s['max']} tokens")
        print(f"    P90     : {s['p90']:.1f} tokens")
        print(f"    < 512   : {s['<512']}  ({s['<512']/n*100:.1f}%)")
        print(f"    > 512   : {s['>512']}  ({s['>512']/n*100:.1f}%)")
        print(f"    > 1024  : {s['>1024']}  ({s['>1024']/n*100:.1f}%)")
        print(f"    > 2048  : {s['>2048']}  ({s['>2048']/n*100:.1f}%)")


def save_csv_report(all_stats: dict, output_path: Path) -> None:
    """将汇总统计保存为 CSV。"""
    rows = []
    for dataset_name, stats in all_stats.items():
        label = stats.get("label") or {}
        for field in ["claim", "doc", "claim+doc"]:
            s = stats.get(field)
            if s is None:
                continue
            rows.append({
                "dataset": dataset_name,
                "col_key": s["col_key"],
                "n_samples": stats["n_samples"],
                "n_neg": label.get("n_neg"),
                "n_pos": label.get("n_pos"),
                "neg_ratio": round(label["neg_ratio"], 4) if "neg_ratio" in label else None,
                "mean": round(s["mean"], 2),
                "min": s["min"],
                "max": s["max"],
                "p90": round(s["p90"], 2),
                "lt_512": s["<512"],
                "gt_512": s[">512"],
                "gt_1024": s[">1024"],
                "gt_2048": s[">2048"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 汇总 CSV 已保存至: {output_path}")


# ─────────────────────────────────────────────────────────────
#  MAIN 配置区（直接在此修改参数，无需外部配置文件）
# ─────────────────────────────────────────────────────────────
def main():
    # ── 模型路径 ──────────────────────────────────────────────
    MODEL_PATH = r"../models/embeddings/Qwen3-Embedding-0.6B"

    # ── 数据根目录 ────────────────────────────────────────────
    DATA_ROOT = r"../data"

    # ── 各数据集的 claim / doc 列名映射 ──────────────────────
    # 格式: "数据集文件夹名": ("claim列名", "doc列名")
    DEFAULT_CLAIM_COL = "claim"
    DEFAULT_DOC_COL = "doc"
    DEFAULT_LABEL_COL = "label"

    DATASET_COL_MAP: dict[str, tuple[str, str]] = {
        # "AggreFact-CNN": ("claim", "doc"),
        # "summeval":      ("claim", "document"),
    }

    # ── 输出 CSV 路径 ─────────────────────────────────────────
    OUTPUT_CSV = r"./token_count_report.csv"

    # ── Tokenization batch size ───────────────────────────────
    BATCH_SIZE = 512

    # ── 设备检测（tokenize 仍在 CPU；此处仅记录可用加速后端）──
    device = resolve_device()
    print(f"[INFO] 当前选用设备标记: {device}（token 统计在 CPU 执行）")

    # ─────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(MODEL_PATH)

    data_root = Path(DATA_ROOT)
    all_stats: dict = {}

    dataset_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"\n[INFO] 发现 {len(dataset_dirs)} 个数据集文件夹。")

    for dataset_dir in dataset_dirs:
        parquet_path = dataset_dir / "processed_data" / "with_id.parquet"
        if not parquet_path.exists():
            print(f"[SKIP] {dataset_dir.name}: 未找到 {parquet_path}")
            continue

        dataset_name = dataset_dir.name
        claim_col, doc_col = DATASET_COL_MAP.get(
            dataset_name, (DEFAULT_CLAIM_COL, DEFAULT_DOC_COL)
        )

        print(f"\n[INFO] 处理数据集: {dataset_name}  ({parquet_path})")
        stats = analyze_dataset(
            parquet_path,
            tokenizer,
            claim_col,
            doc_col,
            label_col=DEFAULT_LABEL_COL,
            batch_size=BATCH_SIZE,
        )
        all_stats[dataset_name] = stats
        print_report(dataset_name, stats)

    save_csv_report(all_stats, Path(OUTPUT_CSV))
    print("\n[DONE] 所有数据集分析完成。")


if __name__ == "__main__":
    main()
