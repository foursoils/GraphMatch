"""
Token Count Analysis for Claim and Doc Fields
使用 Qwen3-Embedding-0.6B 的 tokenizer 统计各数据集中 claim 和 doc 的 token 数分布。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """加载 tokenizer，优先使用 CUDA（若可用）。"""
    print(f"[INFO] 正在加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[INFO] Tokenizer 加载完成，vocab size = {tokenizer.vocab_size}")
    return tokenizer


def count_tokens(texts: list[str], tokenizer: AutoTokenizer, batch_size: int = 512) -> list[int]:
    """批量统计 token 数（纯 tokenizer 操作，无需 GPU）。"""
    counts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  统计 token", leave=False):
        batch = texts[i: i + batch_size]
        # 只 tokenize，不 pad/truncate，获取真实长度
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


def analyze_dataset(
    parquet_path: Path,
    tokenizer: AutoTokenizer,
    claim_col: str,
    doc_col: str,
) -> dict:
    """读取单个数据集并统计 token 分布。"""
    df = pd.read_parquet(parquet_path)

    result = {"n_samples": len(df), "columns": list(df.columns)}

    token_counts_dict = {}
    for col_name, col_key in [("claim", claim_col), ("doc", doc_col)]:
        if col_key not in df.columns:
            print(f"  [WARN] 列 '{col_key}' 不存在，跳过。可用列: {list(df.columns)}")
            result[col_name] = None
            continue

        texts = df[col_key].fillna("").astype(str).tolist()
        token_counts = count_tokens(texts, tokenizer)
        token_counts_dict[col_name] = token_counts

        arr = np.array(token_counts)
        result[col_name] = {
            "col_key": col_key,
            "mean":    float(np.mean(arr)),
            "min":     int(np.min(arr)),
            "max":     int(np.max(arr)),
            "p90":     float(np.percentile(arr, 90)),
            "<512":    int(np.sum(arr < 512)),
            ">512":    int(np.sum(arr > 512)),
            ">1024":   int(np.sum(arr > 1024)),
            ">2048":   int(np.sum(arr > 2048)),
        }

    if "claim" in token_counts_dict and "doc" in token_counts_dict:
        combined_counts = np.array(token_counts_dict["claim"]) + np.array(token_counts_dict["doc"])
        result["claim+doc"] = {
            "col_key": f"{claim_col} + {doc_col}",
            "mean":    float(np.mean(combined_counts)),
            "min":     int(np.min(combined_counts)),
            "max":     int(np.max(combined_counts)),
            "p90":     float(np.percentile(combined_counts, 90)),
            "<512":    int(np.sum(combined_counts < 512)),
            ">512":    int(np.sum(combined_counts > 512)),
            ">1024":   int(np.sum(combined_counts > 1024)),
            ">2048":   int(np.sum(combined_counts > 2048)),
        }

    return result


def print_report(dataset_name: str, stats: dict) -> None:
    """格式化打印单个数据集的统计报告。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  数据集: {dataset_name}")
    print(f"  样本数: {stats['n_samples']}")
    print(f"  数据列: {stats['columns']}")
    print(sep)

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
        for field in ["claim", "doc", "claim+doc"]:
            s = stats.get(field)
            if s is None:
                continue
            rows.append({
                "dataset":   dataset_name,
                "col_key":   s["col_key"],
                "n_samples": stats["n_samples"],
                "mean":      round(s["mean"], 2),
                "min":       s["min"],
                "max":       s["max"],
                "p90":       round(s["p90"], 2),
                "lt_512":    s["<512"],
                "gt_512":    s[">512"],
                "gt_1024":   s[">1024"],
                "gt_2048":   s[">2048"],
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
    # 如果某数据集列名与默认不同，在此单独指定；否则用 DEFAULT。
    DEFAULT_CLAIM_COL = "claim"
    DEFAULT_DOC_COL   = "doc"

    DATASET_COL_MAP: dict[str, tuple[str, str]] = {
        # "AggreFact-CNN": ("claim", "doc"),  # 与默认相同，可注释掉
        # "summeval":      ("claim", "document"),  # 若 doc 列叫 document，在此覆盖
    }

    # ── 输出 CSV 路径 ─────────────────────────────────────────
    OUTPUT_CSV = r"./token_count_report.csv"

    # ── Tokenization batch size ───────────────────────────────
    BATCH_SIZE = 512

    # ── CUDA 提示（tokenizer 本身不使用 GPU，但此处保留选项供后续扩展）──
    import torch
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print(f"[INFO] CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA 不可用，使用 CPU。")

    # ─────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(MODEL_PATH)

    data_root = Path(DATA_ROOT)
    all_stats: dict = {}

    # 遍历所有数据集文件夹
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
        stats = analyze_dataset(parquet_path, tokenizer, claim_col, doc_col)
        all_stats[dataset_name] = stats
        print_report(dataset_name, stats)

    # 汇总并保存 CSV
    save_csv_report(all_stats, Path(OUTPUT_CSV))
    print("\n[DONE] 所有数据集分析完成。")


if __name__ == "__main__":
    main()
